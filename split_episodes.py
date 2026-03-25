from pathlib import Path

import cv2
import pandas as pd


def main() -> None:
    workspace = Path('/Users/josephxu/Desktop/lekiwi')
    csv_path = workspace / 'lekiwi_green_block.csv'
    video_path = workspace / 'file-000.mp4'
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

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video: {video_path}')

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

    print(f'Output directory: {out_dir}')
    print(f'Episodes written: {len(outputs)}')
    print('First 5 outputs:')
    for row in outputs[:5]:
        print(row)
    print('Last output:')
    print(outputs[-1])


if __name__ == '__main__':
    main()
