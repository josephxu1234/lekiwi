#!/usr/bin/env python3
"""Smoke-test Lekiwi dataset adapter and train_omnivla wiring.

This script validates:
1) CSV/MP4 loading and episode-safe sample indexing
2) per-sample contract expected by PaddedCollatorForActionPrediction_Nav_MMN
3) collator output shapes for num_images_in_input=2
4) train_omnivla.py wiring for dataset_name == "lekiwi"

By default it uses lightweight stubs for tokenizers/prompt builder and image transform,
which is enough to validate adapter correctness without downloading model artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import torch

from prismatic.util.data_utils import PaddedCollatorForActionPrediction_Nav_MMN
from prismatic.vla.datasets.lekiwi_dataset import Lekiwi_Dataset

REQUIRED_SAMPLE_KEYS = {
    "pixel_values",
    "pixel_values_goal",
    "input_ids",
    "labels",
    "dataset_name",
    "modality_id",
    "actions",
    "action_select_mask",
    "goal_pose",
    "obj_pose_norm",
    "img_PIL",
    "gimg_PIL",
    "cur_image",
    "goal_image_8",
    "temp_dist",
}


class StubActionTokenizer:
    def __call__(self, action):
        # Keep a fixed 4-token format so train_utils masks are exercised.
        if hasattr(action, "shape") and len(getattr(action, "shape", [])) == 2:
            return ["<A><B><C><D>" for _ in range(action.shape[0])]
        return "<A><B><C><D>"


class StubBaseTokenizer:
    class _Result:
        def __init__(self, input_ids: List[int]):
            self.input_ids = input_ids

    def __call__(self, text, add_special_tokens=True):
        # Short deterministic IDs; enough for label masking checks.
        return StubBaseTokenizer._Result([1, 11, 12, 13, 14, 2])


class StubPromptBuilder:
    def __init__(self, model_family: str):
        self.turns: List[tuple[str, str]] = []

    def add_turn(self, role: str, value: str):
        self.turns.append((role, value))

    def get_prompt(self) -> str:
        return "\n".join([f"{r}: {v}" for r, v in self.turns])


def stub_image_transform(_img):
    # Emulate fused backbones: one image -> 6 channels.
    return torch.zeros((6, 224, 224), dtype=torch.float32)


def check(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def check_wiring(repo_root: Path):
    train_file = repo_root / "vla-scripts" / "train_omnivla.py"
    text = train_file.read_text(encoding="utf-8")

    check(
        "from prismatic.vla.datasets.lekiwi_dataset import Lekiwi_Dataset" in text,
        "train_omnivla.py is missing Lekiwi_Dataset import.",
    )
    check(
        "cfg.dataset_name.lower() == \"lekiwi\"" in text,
        "train_omnivla.py is missing dataset_name == lekiwi routing branch.",
    )


def boundary_violations(ds: Lekiwi_Dataset) -> int:
    violations = 0
    for eidx, local_t in ds.sample_index:
        ep_len = len(ds.episodes[eidx].frame_indices)
        if local_t + ds.action_horizon - 1 >= ep_len:
            violations += 1
    return violations


def summarize_sample(sample: Dict):
    print("sample.actions:", sample["actions"].shape, sample["actions"].dtype)
    print("sample.pixel_values:", sample["pixel_values"].shape, sample["pixel_values"].dtype)
    print("sample.pixel_values_goal:", sample["pixel_values_goal"].shape, sample["pixel_values_goal"].dtype)
    print("sample.cur_image:", sample["cur_image"].shape, sample["cur_image"].dtype)
    print("sample.goal_image_8:", sample["goal_image_8"].shape, sample["goal_image_8"].dtype)
    print("sample.modality_id:", sample["modality_id"])


def validate_samples(ds: Lekiwi_Dataset, sample_indices: Iterable[int]):
    for idx in sample_indices:
        sample = ds[idx]
        missing = REQUIRED_SAMPLE_KEYS - set(sample.keys())
        check(not missing, f"Sample {idx} missing keys: {sorted(missing)}")
        check(sample["actions"].shape == (8, 4), f"Sample {idx} actions must be (8,4)")
        check(sample["pixel_values"].ndim == 3, f"Sample {idx} pixel_values must be 3D")
        check(sample["pixel_values_goal"].ndim == 3, f"Sample {idx} pixel_values_goal must be 3D")
        check(sample["cur_image"].shape[0] % 3 == 0, f"Sample {idx} cur_image channels must be multiple of 3")
        check(sample["goal_image_8"].shape[0] == 3, f"Sample {idx} goal_image_8 first dim must be 3")


def validate_collator(ds: Lekiwi_Dataset):
    collator = PaddedCollatorForActionPrediction_Nav_MMN(
        model_max_length=512,
        pad_token_id=0,
        padding_side="right",
        num_img=2,
    )

    s0 = ds[0]
    s1 = ds[min(1, len(ds) - 1)]
    batch = collator([s0, s1])

    check(batch["pixel_values"].shape[0] == 2, "Collated batch size should be 2")
    check(batch["actions"].shape == (2, 8, 4), "Collated actions shape must be (2,8,4)")
    check(len(batch["modality_id"]) == 2, "Collated modality list should have 2 elements")

    print("batch.pixel_values:", tuple(batch["pixel_values"].shape), batch["pixel_values"].dtype)
    print("batch.actions:", tuple(batch["actions"].shape), batch["actions"].dtype)
    print("batch.goal_pose:", tuple(batch["goal_pose"].shape), batch["goal_pose"].dtype)
    print("batch.cur_image:", tuple(batch["cur_image"].shape), batch["cur_image"].dtype)
    print("batch.goal_image_8:", tuple(batch["goal_image_8"].shape), batch["goal_image_8"].dtype)


def main():
    parser = argparse.ArgumentParser(description="Validate Lekiwi dataset adapter and train wiring.")
    parser.add_argument("--repo_root", type=str, default=".")
    parser.add_argument("--csv_path", type=str, default="../lekiwi_green_block.csv")
    parser.add_argument("--video_path", type=str, default="../file-000.mp4")
    parser.add_argument("--context_size", type=int, default=5)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    csv_path = Path(args.csv_path).resolve()
    video_path = Path(args.video_path).resolve()

    check(csv_path.exists(), f"CSV not found: {csv_path}")
    check(video_path.exists(), f"Video not found: {video_path}")

    check_wiring(repo_root)

    ds = Lekiwi_Dataset(
        csv_path=str(csv_path),
        video_path=str(video_path),
        context_size=args.context_size,
        action_tokenizer=StubActionTokenizer(),
        base_tokenizer=StubBaseTokenizer(),
        image_transform=stub_image_transform,
        prompt_builder_fn=StubPromptBuilder,
    )

    print("dataset_len:", len(ds))
    print("episode_count:", len(ds.episodes))
    print("boundary_violations:", boundary_violations(ds))
    check(len(ds) > 0, "Dataset length must be > 0")
    check(boundary_violations(ds) == 0, "Action chunk crosses episode boundary in at least one sample")

    probe_indices = [0, len(ds) // 2, len(ds) - 1]
    validate_samples(ds, probe_indices)
    summarize_sample(ds[0])
    validate_collator(ds)

    print("VALIDATION_RESULT: PASS")


if __name__ == "__main__":
    main()
