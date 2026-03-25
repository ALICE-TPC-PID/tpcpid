import subprocess

print("Running CI tests...")

result = subprocess.run(
    ["python3", "run/run.py", "--config", "run/ci/ciconfig.json", "--ci-run", "1", "--skip-question", "1"],
    capture_output=True,
    text=True
)

print(result.stdout)
print(result.stderr)
print("Return code:", result.returncode)

exit(result.returncode)