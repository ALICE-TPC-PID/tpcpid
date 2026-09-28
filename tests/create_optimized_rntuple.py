"""Create a slim, regularly clustered RNTuple for ML loader benchmarks."""
import argparse
import json
import os
import time

import ROOT


COLUMNS = [
    "fTPCInnerParam", "fTgl", "fSigned1Pt", "fMass", "fNormMultTPC",
    "fNormNClustersTPC", "fFt0Occ", "fHadronicRate", "fPhi",
    "fTPCSignal", "fInvDeDxExpTPC",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--tree", default="O2tpcskimv0tree")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    ROOT.EnableImplicitMT(args.threads)
    opts = ROOT.RDF.RSnapshotOptions()
    opts.fMode = "RECREATE"
    opts.fOutputFormat = ROOT.RDF.ESnapshotOutputFormat.kRNTuple
    opts.fCompressionAlgorithm = int(ROOT.RCompressionSetting.EAlgorithm.kLZ4)
    opts.fCompressionLevel = 1
    opts.fApproxZippedClusterSize = 16 * 1024 * 1024
    opts.fMaxUnzippedClusterSize = 64 * 1024 * 1024
    opts.fMaxUnzippedPageSize = 1024 * 1024

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    started = time.perf_counter()
    ROOT.RDataFrame(args.tree, args.source).Snapshot("training", args.output, COLUMNS, opts)
    rows = int(ROOT.RDataFrame("training", args.output).Count().GetValue())
    result = {
        "source": os.path.abspath(args.source),
        "output": os.path.abspath(args.output),
        "rows": rows,
        "columns": len(COLUMNS),
        "seconds": time.perf_counter() - started,
        "size_bytes": os.path.getsize(args.output),
        "compression": "LZ4-1",
        "approx_zipped_cluster_mib": 16,
        "max_unzipped_cluster_mib": 64,
    }
    print("CONVERSION_RESULT=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
