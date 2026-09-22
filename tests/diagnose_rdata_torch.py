import json
import os
import ROOT
import torch

COLUMNS = [
    "fTPCInnerParam", "fTgl", "fSigned1Pt", "fMass", "fNormMultTPC",
    "fNormNClustersTPC", "fFt0Occ", "fHadronicRate", "fPhi",
    "fTPCSignal", "fInvDeDxExpTPC", "entry_id",
]

def run(path, eager):
    df = ROOT.RDataFrame("training", path).Define("entry_id", "double(rdfentry_)")
    loader = ROOT.Experimental.ML.RDataLoader(
        df, columns=COLUMNS, batch_size=262144, batches_in_memory=10,
        shuffle=False, drop_remainder=False, load_eager=eager,
    )
    sums = torch.zeros(len(COLUMNS), dtype=torch.float64)
    rows = 0
    samples = []
    for i, values in enumerate(loader.as_torch()):
        rows += len(values)
        sums += torch.nan_to_num(values).double().sum(dim=0)
        if i == 0:
            samples = values[:4, -1].double().tolist()
    return {"eager": eager, "rows": rows, "sums": sums.tolist(), "entry_samples": samples}

path = os.environ["RNTUPLE_PATH"]
print("TORCH_DIAGNOSTIC=" + json.dumps({"columns": COLUMNS, "lazy": run(path, False), "eager": run(path, True)}))
