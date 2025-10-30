import json
import collections

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
        print("--- ", name, " ---")
        print(json.dumps(source, default=str, indent=4, sort_keys=False), "\n")
    return source

def flatten_dict(dictionary, parent_key='', sep='_'):
    items = []
    for key, value in dictionary.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)

def load_dict_to_mem(dict):
    locals().update(dict)