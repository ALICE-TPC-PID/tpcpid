"""Compare complete multi-epoch batch delivery paths for TPC PID training."""
import argparse
import json
import os
import resource
import time

PROCESS_STARTED = time.perf_counter()

import numpy as np
import torch

from framework.neural_network_class.NeuralNetworkClasses.extract_from_root import (
    load_tree,
    r_load_tree,
)


FEATURES = [
    "fTPCInnerParam", "fTgl", "fSigned1Pt", "fMass", "fNormMultTPC",
    "fNormNClustersTPC", "fFt0Occ", "fHadronicRate", "fPhi",
]
TARGETS = ["fTPCSignal", "fInvDeDxExpTPC"]
COLUMNS = FEATURES + TARGETS


def tensor_checksum(values):
    if isinstance(values, tuple):
        return sum(float(torch.nan_to_num(value).sum().item()) for value in values)
    return float(torch.nan_to_num(values).sum().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=(
        "ttree_uproot", "ttree_rdata", "rntuple_rdata", "rntuple_rdata_eager",
        "rntuple_rdata_eager_flat",
    ))
    parser.add_argument("--ttree", required=True)
    parser.add_argument("--rntuple", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=262144)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    setup_started = time.perf_counter()
    if args.case == "ttree_uproot":
        _, array = load_tree(num_workers=args.workers).load(
            args.ttree, use_vars=COLUMNS, key="O2tpcskimv0tree",
            load_latest=True, dtype=np.float32,
        )
        tensor = torch.from_numpy(array)

        def batches():
            for begin in range(0, len(tensor), args.batch_size):
                values = tensor[begin:begin + args.batch_size]
                yield values[:, :len(FEATURES)], values[:, len(FEATURES):]

        rows_expected = len(tensor)
        loader = None
    else:
        import ROOT

        ROOT.EnableImplicitMT(args.workers)
        path = args.ttree if args.case == "ttree_rdata" else args.rntuple
        tree = "O2tpcskimv0tree" if args.case == "ttree_rdata" else "training"
        dataframe = ROOT.RDataFrame(tree, path)
        loader = r_load_tree(
            dataframe,
            columns=COLUMNS,
            target=None if args.case == "rntuple_rdata_eager_flat" else TARGETS,
            batch_size=args.batch_size,
            batches_in_memory=10,
            shuffle=False,
            drop_remainder=False,
            load_eager=args.case in ("rntuple_rdata_eager", "rntuple_rdata_eager_flat"),
        )
        batches = loader.as_torch
        rows_expected = None

    setup_seconds = time.perf_counter() - setup_started
    epoch_seconds = []
    epoch_rows = []
    checksums = []
    for _ in range(args.epochs):
        started = time.perf_counter()
        rows = 0
        checksum = 0.0
        for values in batches():
            rows += len(values[0]) if isinstance(values, tuple) else len(values)
            checksum += tensor_checksum(values)
        epoch_seconds.append(time.perf_counter() - started)
        epoch_rows.append(rows)
        checksums.append(checksum)
        if rows_expected is None:
            rows_expected = rows
        elif rows != rows_expected:
            raise RuntimeError(f"row count changed: expected {rows_expected}, got {rows}")

    result = {
        "case": args.case,
        "epochs": args.epochs,
        "rows_per_epoch": epoch_rows,
        "checksums": checksums,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "import_seconds": setup_started - PROCESS_STARTED,
        "setup_seconds": setup_seconds,
        "epoch_seconds": epoch_seconds,
        "epoch_mean_seconds": sum(epoch_seconds) / len(epoch_seconds),
        "total_seconds": time.perf_counter() - PROCESS_STARTED,
        "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    print("PIPELINE_RESULT=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
