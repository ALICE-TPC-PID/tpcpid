import sys, os, json, subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="configuration.json", help="Path to the configuration file")
args = parser.parse_args()

if not os.path.exists(args.config):
    print(f"Configuration file {args.config} not found. Aborting")
    sys.exit(1)

with open(args.config, 'r') as config_file:
    CONFIG = json.load(config_file)
sys.path.append(CONFIG['settings']['framework'] + "/framework")
from base import *

LOG = logger("Framework")
LOG.welcome_message()

fetch_upstream("origin", path=CONFIG['settings']['framework'])
fetch_upstream("upstream", path=CONFIG['settings']['framework'])
if CONFIG['settings'].get('git', {}).get('checkout', 0) == 1:
    LOG.framework("Checking out the specified git commit/tag/branch...")
    git_config = CONFIG['settings']['git']
    checkout_from_config(
        git_config=git_config,
        path=CONFIG['settings']['framework']
    )
full_git_config(verbose=True, path=CONFIG['settings']['framework'])

CONFIG = add_name_and_path(CONFIG)
can_we_continue() # Ask user if they want to continue with the given configuration, recreates the output folders if necessary
create_folders(CONFIG)
config_path = copy_config(CONFIG)

masterjob_defaults = {
    "partition": "main",
    "time": "60",
    "mem": "150G",
    "cpus-per-task": "16"
}

deep_update(masterjob_defaults, CONFIG.get('masterjob', {}), name="Masterjob settings", verbose=False)
masterjob_defaults["framework_path"] = CONFIG['settings']['framework']
masterjob_defaults["output_path"] = CONFIG["output"]["general"]["path"]

masterjob_defaults = replace_in_dict_keys(masterjob_defaults, '-', '_')

exec_script = f"""#!/bin/bash
#SBATCH --job-name=TPCPID_MASTERJOB
#SBATCH --chdir={masterjob_defaults['framework_path']}
#SBATCH --time={masterjob_defaults['time']}
#SBATCH --mem={masterjob_defaults['mem']}
#SBATCH --partition={masterjob_defaults['partition']}
#SBATCH --cpus-per-task={masterjob_defaults['cpus_per_task']}
#SBATCH --output={masterjob_defaults['output_path']}/run_%j.out
#SBATCH --error={masterjob_defaults['output_path']}/run_%j.err

time python3 {masterjob_defaults['framework_path']}/run/src/run_framework.py --config $1
"""

script_path = os.path.join(masterjob_defaults["framework_path"], 'run/src/RUN_SLURM.sh')
with open(script_path, 'w') as script_file:
    script_file.write(exec_script)

slurm_out = subprocess.check_output(f"sbatch {script_path} {config_path}", shell=True).decode().strip('\n')
LOG.framework(f"TPCPID_MASTERJOB job submitted successfully. Job ID: {slurm_out.split()[-1]}")