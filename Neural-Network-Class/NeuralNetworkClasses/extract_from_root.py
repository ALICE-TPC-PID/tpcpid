import os
import sys
import uproot
import numpy as np

if('tqdm' in sys.modules):
    from tqdm import tqdm


class load_tree():

    def __init__(self, num_workers=1):
        super().__init__()

        self.num_workers = num_workers

    def print_trees(self, path, num_workers=1):
        Tree = uproot.open(path,
                           file_handler=uproot.MultithreadedFileSource,
                           num_workers=num_workers)
        all_ttrees= dict()
        for cls in Tree.items():
            if isinstance(cls[1], uproot.TTree):
                all_ttrees[cls[0]] = cls[1]

        print(all_ttrees)
        return Tree

    def load_internal(self, path, limit=None, use_vars=0, load_latest=True, key=None, verbose=False):

        root_file = uproot.open(path,
                                file_handler=uproot.MultithreadedFileSource,
                                num_workers=self.num_workers)

        entries = list()
        all_ttrees= dict()
        for i, cls in enumerate(root_file.items()):
            if isinstance(cls[1], uproot.TTree):
                all_ttrees[cls[0]] = cls[1]
                entries.append(cls[1].num_entries)

        # Get the list of branch names
        all_branch_names = list(all_ttrees[list(all_ttrees.keys())[0]].keys())
        selected_branch_names = list()
        for branch in all_branch_names:
            if (use_vars):
                if (branch in use_vars):
                    selected_branch_names.append(branch)
            else:
                selected_branch_names.append(branch)

        # Pre-allocate arrays to store branch values
        castable_branches = list()
        for branch_name in selected_branch_names:
            try:
                all_ttrees[list(all_ttrees.keys())[0]][branch_name].interpretation.numpy_dtype
                castable_branches.append(all_ttrees[list(all_ttrees.keys())[0]][branch_name].name)
            except:
                pass

        load_trees = list(all_ttrees.keys())
        if type(key)==bytes:
            key = str(key.decode('utf-8'))

        # If key is specified
        if key:
            new_load_trees = list()
            new_entries = list()
            if type(key) in [str, list, np.ndarray]:
                for tree in load_trees:
                    if key in tree:
                        new_load_trees.append(tree)
                        new_entries.append(all_ttrees[tree].num_entries)
            entries = new_entries
            load_trees = new_load_trees
            del new_entries, new_load_trees

        # Find latest trees
        if load_latest:
            dict_all_trees = dict()
            new_entries = list()
            for elem in load_trees:
                if key:
                    if key not in elem:
                        continue
                if elem.split(";")[0] in dict_all_trees.keys():
                    dict_all_trees[elem.split(";")[0]].append(int(elem.split(";")[1]))
                else:
                    dict_all_trees[elem.split(";")[0]] = [int(elem.split(";")[1])]
            load_trees = list()
            for latest_tree in dict_all_trees.keys():
                full_tree = str(latest_tree) + ";" + str(int(np.max(dict_all_trees[latest_tree])))
                load_trees.append(full_tree)
                new_entries.append(all_ttrees[full_tree].num_entries)
            entries = new_entries
            del new_entries

        # Pre-allocate arrays to store branch values
        branch_value_arrays = [np.empty(np.sum(entries[:limit]), dtype=all_ttrees[list(all_ttrees.keys())[0]][branch_name].interpretation.numpy_dtype) for branch_name in castable_branches]

        # Loop through each branch and fill the pre-allocated array
        for i, tree in enumerate(load_trees):
            if verbose:
                print(tree, "("+str(i+1)+"/"+str(len(load_trees))+")")

            if(('tqdm' in sys.modules) and verbose):
                with tqdm(total=len(castable_branches)) as pbar:
                    for j, branch_name in enumerate(castable_branches):
                        branch_value_arrays[j][int(np.sum(entries[:i])):int(np.sum(entries[:i+1]))] = all_ttrees[tree][branch_name].array(library="np")
                        pbar.update(1)
            else:
                for j, branch_name in enumerate(castable_branches):
                    branch_value_arrays[j][int(np.sum(entries[:i])):int(np.sum(entries[:i+1]))] = all_ttrees[tree][branch_name].array(library="np")

        # Stack the arrays to create a 2D array
        stacked_values = np.array(branch_value_arrays).T

        if verbose:
            # Print branch names and the stacked values
            print("Branch Names:", selected_branch_names)
            print("Shape of stacked values:", np.shape(stacked_values), "\n")

        return np.array(selected_branch_names), stacked_values

    def load(self, path, limit=None, use_vars=0, load_latest=True, key=None, verbose=False):

        if "*" in path:

            current_path = "/" + os.path.join(*path.split("/")[:-1])
            file_name_prefix = path.split("/")[-1].split("*")[0]
            file_name_suffix = path.split("/")[-1].split("*")[1]

            files_in_dir = []
            for (dirpath, dirnames, filenames) in os.walk(current_path):
                for filename in filenames:
                    if ((file_name_prefix in filename) and (file_name_suffix in filename)):
                        files_in_dir.append(os.path.join(dirpath,filename))
                break

            labels, output = None, None
            for i, file in enumerate(files_in_dir):
                if i == 0:
                    labels, output = self.load_internal(path=file, limit=limit, use_vars=use_vars, load_latest=load_latest, key=key, verbose=verbose)
                else:
                    output = np.vstack((output, self.load_internal(path=file, limit=limit, use_vars=use_vars, load_latest=load_latest, key=key, verbose=verbose)[1]))

            return labels, output

        else:

            return self.load_internal(path=path, limit=limit, use_vars=use_vars, load_latest=load_latest, key=key, verbose=verbose)


    def export_to_tree(self, path, labels, data, overwrite=False):

        # file = uproot.recreate(filename,initial_streamers_bytes=20000)
        if "." in path:
            os.makedirs(os.path.join(*(path.split("/"))[:-1]), exist_ok=overwrite)
        else:
            print("Cannot write to file! Path must be full path to file, not directory! Aborting.")
            exit()

        file = uproot.recreate(path)
        writing_data = data.T
        dicts = {}

        for i, key in enumerate(labels.tolist()):
            dicts[key] = writing_data[i]

        file['data_tree'] = dicts
        file.close()

        print("Successfully exported data to TTree!\n")