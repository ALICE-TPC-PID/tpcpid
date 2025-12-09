import json, collections

from .logger import *

LOG = logger(min_severity="DEBUG", task_name="json_functionalities")

def deep_update(source, overrides, name="Dictionary", verbose = True):
    """
    Update a nested dictionary or similar mapping.
    Modify ``source`` in place.
    """
    for key, value in overrides.items():
        if isinstance(value, collections.abc.Mapping) and value:
            returned = deep_update(source.get(key, {}), value, verbose = False)
            source[key] = returned
        else:
            source[key] = overrides[key]
    if verbose:
        LOG.info(f"--- {name} ---")
        LOG.info(json.dumps(source, default=str, indent=4, sort_keys=False))
    return source

def replace_in_dict_keys(d, find, replace):
    """
    Recursively replace occurrences of `find` with `replace` in all dict keys.

    Parameters:
        d (dict): The dictionary to process.
        find (str): The substring to replace.
        replace (str): The replacement substring.

    Returns:
        dict: A new dictionary with updated keys.
    """
    if not isinstance(d, collections.abc.Mapping):
        return d

    new = {}
    for key, value in d.items():
        new_key = key.replace(find, replace)
        if isinstance(value, collections.abc.Mapping):
            new[new_key] = replace_in_dict_keys(value, find, replace)
        elif isinstance(value, list):
            new[new_key] = [
                replace_in_dict_keys(v, find, replace) if isinstance(v, collections.abc.Mapping) else v
                for v in value
            ]
        else:
            new[new_key] = value

    return new