"""Benchmark eager uproot loading against ROOT 6.40 RDataLoader streaming."""
import argparse
import json
import os
import resource
import time

import numpy as np

from framework.neural_network_class.NeuralNetworkClasses.extract_from_root import (
    load_tree,
    r_load_tree,
)


DEFAULT_COLUMNS = [
    "fTPCInnerParam",
    "fTgl",
    "fSigned1Pt",
    "fMass",
    "fNormMultTPC",
    "fNormNClustersTPC",
    "fFt0Occ",
    "fHadronicRate",
    "fPhi",
    "fTPCSignal",
    "fInvDeDxExpTPC",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=("uproot", "rdata"))
    parser.add_argument("path")
    parser.add_argument("--tree", default="O2tpcskimv0tree")
    parser.add_argument("--batch-size", type=int, default=262144)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    started = time.perf_counter()
    setup_started = time.perf_counter()
    checksum = 0.0

    if args.method == "uproot":
        loader = load_tree(num_workers=args.workers)
        _, values = loader.load(
            args.path,
            use_vars=DEFAULT_COLUMNS,
            key=args.tree,
            load_latest=True,
            dtype=np.float32,
        )
        setup_seconds = time.perf_counter() - setup_started
        iteration_started = time.perf_counter()
        rows = len(values)
        batches = (rows + args.batch_size - 1) // args.batch_size
        for begin in range(0, rows, args.batch_size):
            checksum += float(np.nansum(values[begin:begin + args.batch_size]))
        iteration_seconds = time.perf_counter() - iteration_started
    else:
        import ROOT

        ROOT.EnableImplicitMT(args.workers)
        dataframe = ROOT.RDataFrame(args.tree, args.path)
        loader = r_load_tree(
            dataframe,
            columns=DEFAULT_COLUMNS,
            batch_size=args.batch_size,
            batches_in_memory=10,
            shuffle=False,
            drop_remainder=False,
        )
        setup_seconds = time.perf_counter() - setup_started
        iteration_started = time.perf_counter()
        rows = 0
        batches = 0
        for values in loader.as_numpy():
            rows += len(values)
            batches += 1
            checksum += float(np.nansum(values))
        iteration_seconds = time.perf_counter() - iteration_started

    result = {
        "method": args.method,
        "root_file": os.path.abspath(args.path),
        "tree": args.tree,
        "columns": len(DEFAULT_COLUMNS),
        "rows": rows,
        "batches": batches,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "setup_seconds": setup_seconds,
        "iteration_seconds": iteration_seconds,
        "total_seconds": time.perf_counter() - started,
        "checksum": checksum,
        "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    print("BENCHMARK_RESULT=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
