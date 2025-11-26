import json
import os
from datetime import datetime
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from utils.config_tools import (
    add_name_and_path,
    read_config,
    write_config,
    create_folders,
    copy_scripts,
)

config = read_config(path="configuration.json")
config = add_name_and_path(config)
create_folders(config)
write_config(config, path="configuration.json")
copy_scripts(config)