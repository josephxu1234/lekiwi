import csv
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import IGNORE_INDEX, NUM_ACTIONS_CHUNK
import cv2


@dataclass
class EpisodeData:
    episode_id: int
    global_start: int
    frame_indices: List[int]
    actions: np.ndarray


class Lekiwi_Dataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        video_path: str,
        context_size: int,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
        instruction: str = "go to the green block",
        action_horizon: int = NUM_ACTIONS_CHUNK,
        goal_offset: int = NUM_ACTIONS_CHUNK,
        predict_stop_token: bool = True,
    ):
        self.csv_path = Path(csv_path)
        self.video_path = Path(video_path)
        self._source_video_path = self.video_path
        self.context_size = context_size
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder = prompt_builder_fn
        self.instruction = instruction
        self.action_horizon = action_horizon
        self.goal_offset = goal_offset
        self.predict_stop_token = predict_stop_token

        self.episodes = self._load_episodes()
        self.sample_index = self._build_sample_index()
        self._video_capture = None
        self._prepare_video_for_decoding()

        if len(self.sample_index) == 0:
            raise ValueError("No valid lekiwi samples found. Check CSV/video alignment and action horizon.")

    def __len__(self) -> int:
        return len(self.sample_index)

    def _parse_episode_id(self, value: str) -> int:
        if value is None:
            return 0
        return int(float(value))

    def _parse_frame_index(self, value: str) -> int:
        if value is None:
            return 0
        return int(float(value))

    def _parse_action_from_row(self, row: Dict[str, str]) -> np.ndarray:
        x_key, y_key, theta_key = "x.vel", "y.vel", "theta.vel"
        if (
            x_key in row
            and y_key in row
            and theta_key in row
            and row.get(x_key) not in (None, "")
            and row.get(y_key) not in (None, "")
            and row.get(theta_key) not in (None, "")
        ):
            try:
                x = float(row[x_key])
                y = float(row[y_key])
                theta = float(row[theta_key])
                return np.array([x, y, theta], dtype=np.float32)
            except ValueError:
                # Fall through to packed-action parsing.
                pass

        packed = row.get("action")
        if packed is None:
            raise ValueError("CSV must contain x.vel/y.vel/theta.vel or packed action column.")

        # Support both comma-separated and whitespace-separated action arrays.
        # Examples: "[0.1, -0.2, 0.0]" and "[0.1 -0.2 0.]"
        packed_clean = packed.strip().replace("[", " ").replace("]", " ").replace(",", " ")
        values = np.fromstring(packed_clean, sep=" ", dtype=np.float32)
        if values.size < 3:
            raise ValueError(f"Unable to parse action triplet from row: {packed}")
        return np.array(values[:3], dtype=np.float32)

    def _load_episodes(self) -> List[EpisodeData]:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        rows_by_episode: Dict[int, List[Tuple[int, np.ndarray]]] = {}
        with self.csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                episode_id = self._parse_episode_id(row.get("episode_index"))
                frame_idx = self._parse_frame_index(row.get("frame_index"))
                action_xyz = self._parse_action_from_row(row)
                rows_by_episode.setdefault(episode_id, []).append((frame_idx, action_xyz))

        episodes: List[EpisodeData] = []
        global_offset = 0
        for episode_id in sorted(rows_by_episode.keys()):
            episode_rows = sorted(rows_by_episode[episode_id], key=lambda x: x[0])
            frame_indices = [x[0] for x in episode_rows]
            action_xyz = np.stack([x[1] for x in episode_rows], axis=0)

            actions = np.zeros((action_xyz.shape[0], 4), dtype=np.float32)
            actions[:, :3] = action_xyz

            episodes.append(
                EpisodeData(
                    episode_id=episode_id,
                    global_start=global_offset,
                    frame_indices=frame_indices,
                    actions=actions,
                )
            )
            global_offset += len(frame_indices)

        return episodes

    def _build_sample_index(self) -> List[Tuple[int, int]]:
        sample_index: List[Tuple[int, int]] = []
        for eidx, episode in enumerate(self.episodes):
            max_start = len(episode.frame_indices) - self.action_horizon
            for local_t in range(max(0, max_start + 1)):
                sample_index.append((eidx, local_t))
        return sample_index

    def _get_video_codec(self, video_path: Path) -> Optional[str]:
        ffprobe = shutil.which("ffprobe")
        if ffprobe is None:
            return None

        cmd = [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError:
            return None

        codec = result.stdout.strip().lower()
        return codec or None

    def _can_decode_first_frame(self, video_path: Path) -> bool:

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            return False

        ok, frame = cap.read()
        cap.release()
        return bool(ok and frame is not None)

    def _transcode_to_h264(self, input_path: Path, output_path: Path) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError(
                "OpenCV could not decode the input video."
            )

        cmd = [
            ffmpeg,
            "-y",
            "-v",
            "warning",
            "-i",
            str(input_path),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-preset",
            "fast",
            str(output_path),
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Failed to transcode video for OpenCV decode: {input_path}") from exc

    def _prepare_video_for_decoding(self) -> None:
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        codec = self._get_video_codec(self.video_path)
        fallback_video_path = self.video_path.with_name(f"{self.video_path.stem}_h264{self.video_path.suffix}")

        if codec == "av1":
            print(
                f"using H.264 fallback for OpenCV compatibility: {fallback_video_path}"
            )
            if not fallback_video_path.exists() or not self._can_decode_first_frame(fallback_video_path):
                self._transcode_to_h264(self.video_path, fallback_video_path)
            self.video_path = fallback_video_path
            return

        if not self._can_decode_first_frame(self.video_path):
            print(
                f"using H.264 fallback for OpenCV compatibility: {fallback_video_path}"
            )
            if not fallback_video_path.exists() or not self._can_decode_first_frame(fallback_video_path):
                self._transcode_to_h264(self.video_path, fallback_video_path)
            self.video_path = fallback_video_path

    def _get_video_capture(self):
        if self._video_capture is None:
            cap = cv2.VideoCapture(str(self.video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {self.video_path}")
            self._video_capture = cap
        return self._video_capture

    def _read_frame_rgb(self, global_frame_idx: int) -> Image.Image:
        cap = self._get_video_capture()
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(global_frame_idx))
        ok, frame_bgr = cap.read()
        if not ok:
            raise RuntimeError(f"Failed to read frame {global_frame_idx} from {self.video_path}")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    def _resize_norm(self, image_tensor: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        return TF.resize(image_tensor, size)

    def _build_prompt_and_labels(self, actions: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        current_action = actions[0]
        future_actions = actions[1:]
        current_action_string = self.action_tokenizer(current_action)
        future_actions_string = "".join(self.action_tokenizer(future_actions))
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        conversation = [
            {"from": "human", "value": f"What action should the robot take to {self.instruction}?"},
            {"from": "gpt", "value": action_chunk_string},
        ]

        prompt_builder = self.prompt_builder("openvla")
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        input_ids = torch.tensor(self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids)
        labels = input_ids.clone()
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX
        return input_ids, labels

    def __getitem__(self, idx: int):
        eidx, local_t = self.sample_index[idx]
        episode = self.episodes[eidx]

        local_goal = min(local_t + self.goal_offset, len(episode.frame_indices) - 1)
        current_global = episode.global_start + episode.frame_indices[local_t]
        goal_global = episode.global_start + episode.frame_indices[local_goal]

        current_image_pil = self._read_frame_rgb(current_global)
        goal_image_pil = self._read_frame_rgb(goal_global)

        action_chunk = episode.actions[local_t : local_t + self.action_horizon].copy()
        input_ids, labels = self._build_prompt_and_labels(action_chunk)

        pixel_values_current = self.image_transform(current_image_pil)
        pixel_values_goal = self.image_transform(goal_image_pil)

        context_frames = []
        for k in range(self.context_size + 1):
            local_hist = max(0, local_t - (self.context_size - k))
            hist_global = episode.global_start + episode.frame_indices[local_hist]
            hist_pil = self._read_frame_rgb(hist_global)
            context_frames.append(self._resize_norm(TF.to_tensor(hist_pil), (96, 96)))
        cur_image = torch.cat(context_frames, dim=0)
        goal_image_8 = self._resize_norm(TF.to_tensor(goal_image_pil), (96, 96))

        temp_dist = np.array(float(local_goal - local_t), dtype=np.float32)
        goal_pose = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        obj_pose_norm = np.array([0.0, 0.0], dtype=np.float32)
        action_select_mask = np.array(1.0, dtype=np.float32)

        return dict(
            pixel_values=pixel_values_current,
            pixel_values_goal=pixel_values_goal,
            input_ids=input_ids,
            labels=labels,
            dataset_name="lekiwi",
            modality_id=6,
            actions=action_chunk.astype(np.float32),
            action_select_mask=action_select_mask,
            goal_pose=goal_pose,
            obj_pose_norm=obj_pose_norm,
            img_PIL=current_image_pil,
            gimg_PIL=goal_image_pil,
            cur_image=cur_image.numpy().astype(np.float32),
            goal_image_8=goal_image_8.numpy().astype(np.float32),
            temp_dist=temp_dist,
            lan_prompt=self.instruction,
        )
