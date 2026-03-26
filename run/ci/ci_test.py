import subprocess
import sys

print("Running CI tests...")

subprocess.run(
    ["python3", "-u", "run/run.py", "--config", "run/ci/ciconfig.json", "--ci-run", "1", "--skip-question", "1"],
    stderr=subprocess.STDOUT,
    check=True
)

print("CI tests completed successfully.")