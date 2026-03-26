import os
import sys
import uproot
import numpy as np
import pandas as pd

if "tqdm" in sys.modules:
    from tqdm import tqdm


class load_tree:

    def __init__(self, num_workers=1):
        super().__init__()
        self.num_workers = num_workers

    def _open_root(self, path, num_workers=None):
        if num_workers is None:
            num_workers = self.num_workers
        return uproot.open(
            path,
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

    def load_internal(self, path, limit=None, use_vars=0, load_latest=True, key=None, verbose=False, to_numpy=False):
        root_file = self._open_root(path)

        all_ttrees, entries = self._collect_tabular_objects(root_file, verbose=verbose)

        if len(all_ttrees) == 0:
            raise RuntimeError(
                f"No TTree/RNTuple-like objects found in ROOT file: {path}. "
                f"Available keys: {list(root_file.keys())}"
            )

        first_key = list(all_ttrees.keys())[0]
        all_branch_names = list(all_ttrees[first_key].keys())

        selected_branch_names = []
        for branch in all_branch_names:
            if use_vars:
                if branch in use_vars:
                    selected_branch_names.append(branch)
            else:
                selected_branch_names.append(branch)

        if len(selected_branch_names) == 0:
            raise RuntimeError(
                f"No matching branches found in ROOT file: {path}. "
                f"Requested use_vars={use_vars}, available={all_branch_names}"
            )

        load_trees = list(all_ttrees.keys())

        if isinstance(key, bytes):
            key = key.decode("utf-8")

        if key:
            new_load_trees = []
            new_entries = []
            if isinstance(key, (str, list, np.ndarray)):
                for tree in load_trees:
                    if key in tree:
                        new_load_trees.append(tree)
                        new_entries.append(all_ttrees[tree].num_entries)
            load_trees = new_load_trees
            entries = new_entries

        if load_latest:
            load_trees = self._select_latest_cycles(load_trees, all_ttrees, key_filter=key)
            entries = [all_ttrees[t].num_entries for t in load_trees]

        if len(load_trees) == 0:
            raise RuntimeError(
                f"No trees selected from ROOT file: {path}. "
                f"Available objects: {list(all_ttrees.keys())}, key filter={key}"
            )

        frames = []

        for i, tree in enumerate(load_trees):
            if verbose:
                print(f"{tree} ({i+1}/{len(load_trees)})")

            obj = all_ttrees[tree]

            try:
                # Read all selected columns at once. Works much better for both TTree and RNTuple.
                arrs = obj.arrays(selected_branch_names, library="np")

                if limit is not None:
                    arrs = {k: v[:limit] for k, v in arrs.items()}

                dfnew = pd.DataFrame(arrs)

                missing = [b for b in selected_branch_names if b not in dfnew.columns]
                if missing:
                    raise RuntimeError(
                        f"Missing requested branches in {tree}: {missing}. "
                        f"Available: {list(obj.keys())}"
                    )

                # Preserve requested column order
                dfnew = dfnew[selected_branch_names]
                frames.append(dfnew)

            except Exception as e:
                raise RuntimeError(
                    f"Failed to read branches from object '{tree}' in file '{path}'. "
                    f"Requested branches: {selected_branch_names}. "
                    f"Original error: {e}"
                ) from e

        if len(frames) == 0:
            raise RuntimeError(
                f"No data frames were created from file: {path}. "
                f"Selected objects: {load_trees}"
            )

        df = pd.concat(frames, ignore_index=True)

        if verbose:
            print("Branch Names:", selected_branch_names)
            print("Shape of stacked values:", df.shape, "\n")

        if to_numpy:
            return np.array(selected_branch_names), df.to_numpy()
        else:
            return np.array(selected_branch_names), df

    def load(self, path, limit=None, use_vars=0, load_latest=True, key=None, verbose=False):
        if "*" in path:
            current_path = "/" + os.path.join(*path.split("/")[:-1])
            file_name_prefix = path.split("/")[-1].split("*")[0]
            file_name_suffix = path.split("/")[-1].split("*")[1]

            files_in_dir = []
            for (dirpath, dirnames, filenames) in os.walk(current_path):
                for filename in filenames:
                    if (file_name_prefix in filename) and (file_name_suffix in filename):
                        files_in_dir.append(os.path.join(dirpath, filename))
                break

            labels, output = None, None
            for i, file in enumerate(files_in_dir):
                if i == 0:
                    labels, output = self.load_internal(
                        path=file,
                        limit=limit,
                        use_vars=use_vars,
                        load_latest=load_latest,
                        key=key,
                        verbose=verbose,
                    )
                else:
                    output = pd.concat(
                        [
                            output,
                            self.load_internal(
                                path=file,
                                limit=limit,
                                use_vars=use_vars,
                                load_latest=load_latest,
                                key=key,
                                verbose=verbose,
                            )[1],
                        ],
                        ignore_index=True,
                    )

            return labels, output.to_numpy()

        else:
            return self.load_internal(
                path=path,
                limit=limit,
                use_vars=use_vars,
                load_latest=load_latest,
                key=key,
                verbose=verbose,
                to_numpy=True,
            )

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