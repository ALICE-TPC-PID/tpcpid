"""
File: create_training_data.py
Author: Christian Sonnabend
Email: christian.sonnabend@cern.ch
Date: 15/03/2024
"""

import json
import sys
import os
import argparse
import subprocess

### Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-cd", "--current-dir", default=".", help="Directory for training of the neural network")
args = parser.parse_args()

current_dir = str(args.current_dir)

### External json settings
configs_file = open("config.json", "r")
CONF = json.load(configs_file)