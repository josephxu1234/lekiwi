from pathlib import Path
import shutil
import subprocess

import cv2
import pandas as pd


def _get_video_codec(video_path: Path) -> str | None:
    ffprobe = shutil.which('ffprobe')
    if ffprobe is None:
        return None

    cmd = [
        ffprobe,
        '-v',
        'error',
        '-select_streams',
        'v:0',
        '-show_entries',
        'stream=codec_name',
        '-of',
        'default=noprint_wrappers=1:nokey=1',
        str(video_path),
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        return None

    codec = result.stdout.strip().lower()
    return codec or None


def _can_decode_first_frame(video_path: Path) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return False

    ok, frame = cap.read()
    cap.release()
    return bool(ok and frame is not None)


def _transcode_to_h264(input_path: Path, output_path: Path) -> None:
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        raise RuntimeError(
            'OpenCV failed to decode the input video.'
        )

    cmd = [
        ffmpeg,
        '-y',
        '-v',
        'warning',
        '-i',
        str(input_path),
        '-an',
        '-c:v',
        'libx264',
        '-pix_fmt',
        'yuv420p',
        '-crf',
        '18',
        '-preset',
        'fast',
        str(output_path),
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f'Failed to transcode {input_path} to H.264 using ffmpeg.'
        ) from exc


def main() -> None:
    workspace = Path('/scratch/gpfs/TSILVER/jx6/lekiwi')
    csv_path = workspace / 'lekiwi_green_block.csv'
    video_path = workspace / 'file-000.mp4'
    fallback_video_path = workspace / 'file-000_h264.mp4'
    out_dir = workspace / 'episode_videos'
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    required = {'episode_index', 'frame_index'}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f'Missing required columns: {sorted(missing)}')

    df['episode_index'] = pd.to_numeric(df['episode_index'], errors='raise').astype(int)
    df['frame_index'] = pd.to_numeric(df['frame_index'], errors='raise').astype(int)
    df = df.reset_index(drop=True)

    for ep, g in df.groupby('episode_index', sort=False):
        fi = g['frame_index'].to_numpy()
        if fi[0] != 0:
            raise RuntimeError(
                f'Episode {ep}: frame_index does not start at 0 (got {fi[0]}).'
            )
        contiguous = ((fi[1:] - fi[:-1]) == 1).all() if len(fi) > 1 else True
        if not contiguous:
            raise RuntimeError(f'Episode {ep}: frame_index is not contiguous.')

    active_video_path = video_path
    codec = _get_video_codec(video_path)
    if codec == 'av1':
        print(
            f'Transcoding to H.264: {fallback_video_path}'
        )
        if not fallback_video_path.exists() or not _can_decode_first_frame(fallback_video_path):
            _transcode_to_h264(video_path, fallback_video_path)
        active_video_path = fallback_video_path
    elif not _can_decode_first_frame(video_path):
        print(
            f'Transcoding to H.264: {fallback_video_path}'
        )
        if not fallback_video_path.exists() or not _can_decode_first_frame(fallback_video_path):
            _transcode_to_h264(video_path, fallback_video_path)
        active_video_path = fallback_video_path

    cap = cv2.VideoCapture(str(active_video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video: {active_video_path}')

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames != len(df):
        raise RuntimeError(
            f'Video frame count ({total_frames}) does not match CSV rows ({len(df)}).'
        )

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    start_global = 0
    outputs = []
    grouped = df.groupby('episode_index', sort=False)
    for ep, g in grouped:
        n = len(g)
        end_global = start_global + n - 1
        out_path = out_dir / f'episode_{ep:03d}.mp4'

        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f'Could not open writer for {out_path}')

        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_global))
        for _ in range(n):
            ok, frame = cap.read()
            if not ok or frame is None:
                writer.release()
                cap.release()
                raise RuntimeError(
                    f'Failed reading frame while writing episode {ep}'
                )
            writer.write(frame)

        writer.release()
        outputs.append((ep, n, start_global, end_global, out_path.name))
        start_global = end_global + 1

    cap.release()

    print(f'Input video used: {active_video_path}')
    print(f'Output directory: {out_dir}')
    print(f'Episodes written: {len(outputs)}')
    print('First 5 outputs:')
    for row in outputs[:5]:
        print(row)
    print('Last output:')
    print(outputs[-1])


if __name__ == '__main__':
    main()
