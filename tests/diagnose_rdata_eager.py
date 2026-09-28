"""Diagnose ROOT 6.40 RDataLoader eager-mode row/value differences."""
import json
import os

import numpy as np
import ROOT


COLUMNS = [
    "fTPCInnerParam", "fTgl", "fSigned1Pt", "fMass", "fNormMultTPC",
    "fNormNClustersTPC", "fFt0Occ", "fHadronicRate", "fPhi",
    "fTPCSignal", "fInvDeDxExpTPC", "entry_id",
]


def summarize(path, eager):
    dataframe = ROOT.RDataFrame("training", path).Define("entry_id", "double(rdfentry_)")
    loader = ROOT.Experimental.ML.RDataLoader(
        dataframe,
        columns=COLUMNS,
        batch_size=262144,
        batches_in_memory=10,
        shuffle=False,
        drop_remainder=False,
        load_eager=eager,
    )
    sums = np.zeros(len(COLUMNS), dtype=np.float64)
    mins = np.full(len(COLUMNS), np.inf)
    maxs = np.full(len(COLUMNS), -np.inf)
    zeros = np.zeros(len(COLUMNS), dtype=np.int64)
    rows = 0
    batches = []
    entry_discontinuities = 0
    last_entry = None
    entry_samples = []
    for index, values in enumerate(loader.as_numpy()):
        values = np.asarray(values)
        rows += len(values)
        sums += np.nansum(values, axis=0, dtype=np.float64)
        mins = np.minimum(mins, np.nanmin(values, axis=0))
        maxs = np.maximum(maxs, np.nanmax(values, axis=0))
        zeros += np.count_nonzero(values == 0, axis=0)
        ids = values[:, -1].astype(np.int64)
        entry_discontinuities += int(np.count_nonzero(np.diff(ids) != 1))
        if last_entry is not None and ids[0] != last_entry + 1:
            entry_discontinuities += 1
        last_entry = int(ids[-1])
        if index < 3 or len(values) != 262144:
            batches.append({"batch": index, "rows": len(values),
                            "first_entry": int(ids[0]), "last_entry": int(ids[-1])})
        if index == 0:
            entry_samples = ids[:16].tolist()
    return {
        "eager": eager, "rows": rows, "sums": sums.tolist(),
        "mins": mins.tolist(), "maxs": maxs.tolist(), "zeros": zeros.tolist(),
        "entry_discontinuities": entry_discontinuities,
        "entry_samples": entry_samples, "selected_batches": batches,
    }


def main():
    path = os.environ["RNTUPLE_PATH"]
    lazy = summarize(path, False)
    eager = summarize(path, True)
    print("EAGER_DIAGNOSTIC=" + json.dumps({"columns": COLUMNS, "lazy": lazy, "eager": eager}), flush=True)


if __name__ == "__main__":
    main()
