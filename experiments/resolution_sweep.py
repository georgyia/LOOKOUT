"""Resolution sweep for research note 03.

Downscales a tile crop in steps, runs the MediaPipe observer, and reports iris
jitter versus crop size to locate the resolution below which landmarks are
unreliable. Outputs are gitignored; record the resulting numbers in
docs/research/03-landmarks.md.

Usage:
    python experiments/resolution_sweep.py CLIP.mp4 MODEL.task
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path


def main(clip: str, model: str) -> None:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - experiment-only dependency
        raise SystemExit("install the 'cv' extra: pip install -e '.[cv]'") from exc

    from lookout.face import MediaPipeFaceObserver
    from lookout.frames import read_video

    observer = MediaPipeFaceObserver(model)
    sizes = [256, 192, 160, 128, 96, 64, 48]

    for size in sizes:
        xs: list[float] = []
        ys: list[float] = []
        for frame in read_video(clip, target_fps=5.0):
            crop = cv2.resize(frame.image, (size, size))
            observation = observer.observe(crop)
            if observation is None:
                continue
            ix, iy = observation.right_iris
            xs.append(ix)
            ys.append(iy)
        if len(xs) < 2:
            print(f"{size:>4}px: no stable detection")
            continue
        jitter = statistics.pstdev(xs) + statistics.pstdev(ys)
        print(f"{size:>4}px: n={len(xs):3d} iris_jitter={jitter:.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 3 or not Path(sys.argv[1]).exists():
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
