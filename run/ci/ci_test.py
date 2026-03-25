import subprocess

print("Running CI tests...")

result = subprocess.run(
    ["python3", "run/run.py", "--config", "ciconfig.json"],
    capture_output=True,
    text=True
)

print(result.stdout)
print(result.stderr)
print("Return code:", result.returncode)