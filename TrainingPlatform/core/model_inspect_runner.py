"""Read a checkpoint in the user's ML environment and emit plain JSON."""
import argparse
import json

if __package__:
    from .worker_runtime import prepare_runner_paths
else:
    from worker_runtime import prepare_runner_paths


def main(argv=None):
    prepare_runner_paths(__file__)
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    args = parser.parse_args(argv)
    from TrainingPlatform.core.checkpoint_metadata import read_checkpoint
    try:
        result = read_checkpoint(args.weights)
        print("__MODEL_METADATA__" + json.dumps(result, ensure_ascii=False, default=str), flush=True)
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
