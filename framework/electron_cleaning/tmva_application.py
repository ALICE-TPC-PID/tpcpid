"""
Apply TMVA BDT selection to multiple TTrees stored in subdirectories of a ROOT file.

Example input structure:
    input.root
    ├── DF_1/O2tpcskimv0wde (TTree)
    ├── DF_2/O2tpcskimv0wde
    └── ...

Output structure:
    selected_events.root
    ├── DF_1/O2tpcskimv0wde (filtered)
    ├── DF_2/O2tpcskimv0wde
    └── ...

Usage:
    python tmva_application.py [input_file] [output_file] [weight_file]
"""

import ROOT
from array import array
import os, sys, json
import argparse


def apply_tmva_to_directories(
    input_file_path,
    output_file_path,
    weight_file_path
):

    input_file = ROOT.TFile.Open(input_file_path)
    if not input_file or input_file.IsZombie():
        LOG.fatal(f"Cannot open input file '{input_file_path}'.")
        return

    output_file = ROOT.TFile(output_file_path, "RECREATE")
    if not output_file or output_file.IsZombie():
        LOG.fatal(f"Cannot create output file '{output_file_path}'.")
        input_file.Close()
        return

    if not os.path.exists(weight_file_path):
        LOG.fatal(f"TMVA weight file not found at '{weight_file_path}'.")
        input_file.Close()
        output_file.Close()
        return

    # TMVA Reader
    reader = ROOT.TMVA.Reader("!Color:!Silent")

    # Variables required by TMVA (must match training names)
    fGammaPsiPair = array("f", [0.0])
    fAlphaV0 = array("f", [0.0])
    fCosPAV0 = array("f", [0.0])
    fNSigTPC = array("f", [0.0])
    fTPCSignal = array("f", [0.0])
    fTPCInnerParam = array("f", [0.0])
    fPidIndex = array("b", [0])

    reader.AddVariable("fGammaPsiPair", fGammaPsiPair)
    reader.AddVariable("fAlphaV0", fAlphaV0)
    reader.AddVariable("fCosPAV0", fCosPAV0)

    reader.AddSpectator("1./fTPCInnerParam", fTPCInnerParam)
    reader.AddSpectator("fNSigTPC", fNSigTPC)
    reader.AddSpectator("fTPCSignal", fTPCSignal)

    LOG.info(f"Loading TMVA weights: {weight_file_path}")
    reader.BookMVA("BDT", weight_file_path)

    # Loop through directories
    for key in input_file.GetListOfKeys():
        dir_name = key.GetName()
        obj = key.ReadObj()

        if not obj.InheritsFrom("TDirectory"):
            continue

        LOG.info(f"\nProcessing directory: {dir_name}")

        input_tree = obj.Get("O2tpcskimv0wde")
        input_tree2 = obj.Get("O2tpctofskimwde")
        if not input_tree or not isinstance(input_tree, ROOT.TTree):
            LOG.warning(f"TTree 'O2tpcskimv0wde' missing in '{dir_name}'. Skipping.")
            continue

        output_dir = output_file.mkdir(dir_name)
        output_dir.cd()

        output_tree = input_tree.CloneTree(0)
        output_tree2 = input_tree2.CloneTree()
        output_tree.SetDirectory(output_dir)
        output_tree2.SetDirectory(output_dir)

        n_entries = input_tree.GetEntries()
        LOG.info(f"➤ Entries: {n_entries}")

        selected_count = 0

        # Event loop
        for i, event in enumerate(input_tree):
            if i % max(1, n_entries // 10) == 0:
                LOG.info(f"  - {i}/{n_entries} processed")

            try:
                fGammaPsiPair[0] = float(event.fGammaPsiPair)
                fAlphaV0[0] = float(event.fAlphaV0)
                fCosPAV0[0] = float(event.fCosPAV0)
                fNSigTPC[0] = float(event.fNSigTPC)
                fTPCSignal[0] = float(event.fTPCSignal)

                raw = event.fPidIndex
                fPidIndex[0] = raw[0] if isinstance(raw, bytes) else ord(raw) if isinstance(raw, str) else int(raw)

                fTPCInnerParam[0] = 1.0 / float(event.fTPCInnerParam) if event.fTPCInnerParam != 0 else 0.0

            except AttributeError as e:
                LOG.error(f"Missing branch at entry {i}: {e}")
                continue

            # Non-electron events are copied without BDT cut
            if fPidIndex[0] != 0:
                output_tree.Fill()
                continue

            bdt_response = reader.EvaluateMVA("BDT")

            # Apply selection
            if bdt_response > 0.0 or fTPCInnerParam[0] > 2.0:
                output_tree.Fill()
                selected_count += 1

        LOG.info(f"Selected: {selected_count} entries.")
        output_file.cd()
        output_dir.Write()
    output_file.Close()
    input_file.Close()

    LOG.info(f"\n🎉 Finished. Output saved to: {output_file_path}")


if __name__ == "__main__":
    argc = len(sys.argv)
    
    parser = argparse.ArgumentParser(description="Apply TMVA BDT selection to ROOT TTrees in subdirectories.")
    parser.add_argument("--config", type=str, default="configuration.json", help="Path to configuration file")
    args = parser.parse_args()
    
    config = args.config
    with open(config, 'r') as config_file:
        CONFIG = json.load(config_file)
    sys.path.append(CONFIG['settings']['framework'] + "/framework")
    from base import *
    LOG = logger(min_severity=CONFIG["process"].get("severity", "DEBUG"), task_name="tmva_application")
    
    CONFIG["output"]["electronCleaning"]["tmva_training_input_path"] = CONFIG["dataset"]["input_skimmedtree_path"]
    CONFIG["output"]["electronCleaning"]["tmva_training_output_path"] = CONFIG["output"]["general"]["trees"] + "/tmva_electron_cleaned.root"
    CONFIG["output"]["electronCleaning"]["tmva_training_weights_path"] = os.path.join(CONFIG["output"]["general"]["path"], "electronCleaning/bdt/weights/TMVAClassification_BDT.weights.xml") #CONFIG["output"]["electronCleaning"]["path"] + "/TMVAClassification_BDT.weights.xml"
    
    write_config(CONFIG, path=args.config)

    apply_tmva_to_directories(
        input_file_path=CONFIG["output"]["electronCleaning"]["tmva_training_input_path"],
        output_file_path=CONFIG["output"]["electronCleaning"]["tmva_training_output_path"],
        weight_file_path=CONFIG["output"]["electronCleaning"]["tmva_training_weights_path"]
    )
