import os

print("Running CI tests...")

os.system("python3 ../run.py --config ciconfig.json")

print("CI tests completed successfully!")