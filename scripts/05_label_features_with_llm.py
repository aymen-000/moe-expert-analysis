import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from moe_interp.models import make_dirs
from moe_interp.autointerp import label_features_with_llm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="./moe_interp_out")
    parser.add_argument("--model-id", default="gemini-2.5-flash")
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument("--sleep-s", type=float, default=0.5)
    args = parser.parse_args()

    dirs = make_dirs(args.base_dir)
    prompts_path = os.path.join(dirs["autointerp"], "all_prompts.jsonl")
    out_path = os.path.join(dirs["autointerp"], "labeled_features.jsonl")

    label_features_with_llm(
        prompts_path, out_path,
        model_id=args.model_id, max_features=args.max_features, sleep_s=args.sleep_s,
    )


if __name__ == "__main__":
    main()
