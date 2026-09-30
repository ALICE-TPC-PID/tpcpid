"""Train all CI networks; optionally export them for the O2Physics job."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", help="New directory for CI network artifacts")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    command = [sys.executable, "-u", "run/run.py", "--config", "run/ci/ciconfig.json",
               "--ci-run", "1", "--skip-question", "1"]
    if args.artifact_dir:
        artifact_dir = Path(args.artifact_dir).resolve()
        if artifact_dir.exists():
            parser.error(f"Artifact directory already exists: {artifact_dir}")
        command.extend(["--ci-artifact-dir", str(artifact_dir)])
    print("Running CI tests...", flush=True)
    subprocess.run(command, cwd=repo, stderr=subprocess.STDOUT, check=True)
    print("CI tests completed successfully.", flush=True)


if __name__ == "__main__":
    main()
