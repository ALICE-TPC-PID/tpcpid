import subprocess

print("Running CI tests...")

result = subprocess.run(
    ["python3", "run/run.py", "--config", "ciconfig.json", "--ci", "1"],
    check=True  # this will raise if it fails
)

print("CI tests completed successfully!")