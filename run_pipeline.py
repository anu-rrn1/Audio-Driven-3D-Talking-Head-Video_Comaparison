"""Run the steps of a pipeline in order.

python run_pipeline.py flame                     # all FLAME pipeline steps
python run_pipeline.py flame --from 05 --to 07   # a range of steps
python run_pipeline.py flame --skip 04a 04b      # e.g. reuse the trained audio model
python run_pipeline.py baseline --synthetic      # video baseline on synthetic data
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent / "scripts"
PIPELINES = {"flame": SCRIPTS / "flame_pipeline", "baseline": SCRIPTS / "video_baseline"}


def step_id(script: Path) -> str:
    return script.name.split("_")[0]


def select_steps(pipeline, first, last, skip, synthetic):
    steps = sorted(PIPELINES[pipeline].glob("[0-9]*.py"))
    if pipeline == "baseline":  # step 00 replaces step 01 when no real video is used
        steps = [s for s in steps if step_id(s) != ("01" if synthetic else "00")]
    return [
        s
        for s in steps
        if (first is None or step_id(s) >= first)
        and (last is None or step_id(s) <= last)
        and step_id(s) not in skip
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("pipeline", choices=sorted(PIPELINES))
    parser.add_argument("--from", dest="first", help="first step id, e.g. 02")
    parser.add_argument("--to", dest="last", help="last step id, e.g. 05")
    parser.add_argument("--skip", nargs="*", default=[], help="step ids to skip, e.g. 04a 04b")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="baseline only: use the synthetic video instead of data/raw/video.mp4",
    )
    args = parser.parse_args()

    for script in select_steps(
        args.pipeline, args.first, args.last, set(args.skip), args.synthetic
    ):
        print(f"\n===== {script.relative_to(SCRIPTS)} =====", flush=True)
        subprocess.run([sys.executable, str(script)], check=True)
    print("\nDone. Results are in outputs/.")


if __name__ == "__main__":
    main()
