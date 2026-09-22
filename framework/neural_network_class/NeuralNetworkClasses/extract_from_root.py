import os
import glob
import sys
import uproot
import numpy as np
import pandas as pd

if "tqdm" in sys.modules:
    from tqdm import tqdm


class r_load_tree:
    """Lazily construct ROOT 6.40's experimental ML ``RDataLoader``.

    PyROOT is deliberately imported only when this class is instantiated. This
    keeps importing this module working in uproot-only environments and on
    developer machines without ROOT. The returned object is ROOT's loader
    itself and therefore follows its experimental API.
    """

    def __new__(cls, *args, **kwargs):
        try:
            import ROOT
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "r_load_tree requires PyROOT with "
                "ROOT.Experimental.ML.RDataLoader (ROOT 6.40 or newer)"
            ) from exc

        try:
            loader = ROOT.Experimental.ML.RDataLoader
        except AttributeError as exc:
            version = getattr(ROOT, "__version__", "unknown")
            raise RuntimeError(
                "r_load_tree requires ROOT.Experimental.ML.RDataLoader "
                f"(ROOT 6.40 or newer); found ROOT {version}"
            ) from exc

        return loader(*args, **kwargs)


class load_tree:

    def __init__(self, num_workers=1):
        super().__init__()
        self.num_workers = num_workers

    def _open_root(self, path, num_workers=None):
        if num_workers is None:
            num_workers = self.num_workers
        return uproot.open(
            path,
            array_cache=None,
            file_handler=uproot.MultithreadedFileSource,
            num_workers=num_workers,
        )

    def _is_tabular_object(self, obj):
        return (
            isinstance(obj, uproot.TTree)
            or (
                hasattr(obj, "keys")
                and hasattr(obj, "arrays")
                and hasattr(obj, "num_entries")
            )
        )

    def _collect_tabular_objects(self, root_file, verbose=False):
        all_objs = {}
        entries = []

        if verbose:
            print(f"[DEBUG] Reading ROOT file")
            print(f"[DEBUG] Top-level keys: {list(root_file.keys())}")

        for key, obj in root_file.items():
            if verbose:
                print(f"[DEBUG] Key: {key}, type: {type(obj)}")

            if self._is_tabular_object(obj):
                all_objs[key] = obj
                entries.append(obj.num_entries)
                if verbose:
                    print(f"[DEBUG] -> accepted as tabular object, entries={obj.num_entries}")

        return all_objs, entries

    def _select_latest_cycles(self, load_keys, all_objs, key_filter=None):
        """
        For classic ROOT cycle keys like 'data_tree;1', keep only the latest cycle.
        For keys without ';N', keep them as-is.
        """
        if not load_keys:
            return []

        grouped = {}
        passthrough = []

        for elem in load_keys:
            if key_filter and key_filter not in elem:
                continue

            if ";" in elem:
                base, cycle = elem.rsplit(";", 1)
                try:
                    cycle = int(cycle)
                    grouped.setdefault(base, []).append(cycle)
                except ValueError:
                    passthrough.append(elem)
            else:
                passthrough.append(elem)

        out = []
        for base, cycles in grouped.items():
            out.append(f"{base};{max(cycles)}")
        out.extend(passthrough)

        return out

    def trees(self, path, num_workers=1):
        root_file = self._open_root(path, num_workers=num_workers)
        all_objs, _ = self._collect_tabular_objects(root_file, verbose=False)

        out = {}
        for key, obj in all_objs.items():
            out[key] = list(obj.keys())

        return out

    def load_internal(self, path, limit=None, use_vars=0, load_latest=True,
                      key=None, verbose=False, to_numpy=False, dtype=None,
                      step_size="64 MB"):
        """Read selected scalar branches into one array using bounded ROOT chunks.

        limit is a per-tree entry limit, as in the original API, but now limits
        disk reads too. dtype=None preserves the common source dtype.
        """
        if limit is not None and (int(limit) != limit or limit < 0):
            raise ValueError('limit must be a nonnegative integer or None')
        with self._open_root(path) as root_file:
            objects, _ = self._collect_tabular_objects(root_file, verbose=verbose)
            if isinstance(key, bytes):
                key = key.decode('utf-8')
            filters = [key] if isinstance(key, str) else key
            selected = [name for name in objects
                        if filters is None or any(part in name for part in filters)]
            if load_latest:
                selected = self._select_latest_cycles(selected, objects)
            if not selected:
                raise RuntimeError(f'No trees selected from {path}; key filter={key}')
            available = list(objects[selected[0]].keys())
            requested = list(use_vars) if not isinstance(use_vars, (int, type(None))) else available
            if not requested:
                raise ValueError('No branches requested')
            for name in selected:
                missing = set(requested) - set(objects[name].keys())
                if missing:
                    raise RuntimeError(f'Missing branches in {name}: {sorted(missing)}')
            counts = [min(objects[name].num_entries, int(limit)) if limit is not None
                      else objects[name].num_entries for name in selected]
            total = sum(counts)
            output, offset = None, 0
            for name, count in zip(selected, counts):
                obj = objects[name]
                remaining = count
                if not remaining:
                    continue
                for arrays in obj.iterate(filter_name=requested, entry_stop=count,
                                          step_size=step_size, library='np'):
                    # TTree returns a dict; RNTuple may return a structured ndarray.
                    columns = [arrays[branch][:remaining] for branch in requested]
                    if any(column.ndim != 1 or column.dtype.kind == 'O' for column in columns):
                        raise ValueError('Only scalar numeric ROOT branches are supported')
                    if output is None:
                        out_dtype = dtype if dtype is not None else np.result_type(*[c.dtype for c in columns])
                        output = np.empty((total, len(requested)), dtype=out_dtype)
                    elif dtype is None:
                        # Preserve NumPy/pandas promotion across differently typed trees.
                        common = np.result_type(output.dtype, *[c.dtype for c in columns])
                        if common != output.dtype:
                            output = output.astype(common)
                    rows = len(columns[0])
                    for i, column in enumerate(columns):
                        output[offset:offset+rows, i] = column
                    offset += rows
                    remaining -= rows
                    if remaining == 0:
                        break
            if output is None:
                output = np.empty((0, len(requested)), dtype=dtype or np.float64)
            if offset != total:
                raise RuntimeError(f'Expected {total} ROOT entries, read {offset}')
        if verbose:
            print('Branches:', requested, 'shape:', output.shape)
        return np.asarray(requested), output if to_numpy else pd.DataFrame(output, columns=requested)

    def load(self, path, limit=None, use_vars=0, load_latest=True, key=None,
             verbose=False, dtype=None, step_size="64 MB"):
        paths = sorted(glob.glob(path)) if glob.has_magic(path) else [path]
        if not paths:
            raise FileNotFoundError(f'No ROOT files match {path}')
        arrays, labels = [], None
        for filename in paths:
            current_labels, values = self.load_internal(filename, limit=limit,
                use_vars=use_vars, load_latest=load_latest, key=key, verbose=verbose,
                to_numpy=True, dtype=dtype, step_size=step_size)
            if labels is not None and not np.array_equal(labels, current_labels):
                raise ValueError('ROOT files have inconsistent branch order/schema')
            labels = current_labels
            arrays.append(values)
        return labels, arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=0)

    def export_to_tree(self, path, labels, data, overwrite=False):
        print("[DEBUG] export path:", path)
        print("[DEBUG] abs export path:", os.path.abspath(path))
        print("[DEBUG] labels:", labels)
        print("[DEBUG] labels.tolist():", labels.tolist())
        print("[DEBUG] data.shape:", data.shape)
        print("[DEBUG] writing_data.shape:", data.T.shape)
        print("[DEBUG] number of branches to write:", len(labels.tolist()))

        if not os.path.isabs(path):
            raise ValueError(f"Path must be absolute, got: {path}")

        dirpath = os.path.dirname(path)
        os.makedirs(dirpath, exist_ok=True)

        if os.path.exists(path) and not overwrite:
            raise FileExistsError(f"File already exists and overwrite=False: {path}")

        file = uproot.recreate(path)
        writing_data = data.T
        dicts = {}

        for i, key in enumerate(labels.tolist()):
            print(f"[DEBUG] branch {i}: name={key}, shape={writing_data[i].shape}, dtype={writing_data[i].dtype}")
            dicts[key] = writing_data[i]

        print("[DEBUG] dict keys:", list(dicts.keys()))
        print("[DEBUG] dict empty:", len(dicts) == 0)

        file["data_tree"] = dicts
        file.close()

        print("[DEBUG] wrote file, exists:", os.path.exists(path))
        print("[DEBUG] file size:", os.path.getsize(path))

        f = uproot.open(path)
        print("[DEBUG] top-level keys after write:", list(f.keys()))
        for k, obj in f.items():
            print("[DEBUG] key:", k, "type:", type(obj))
