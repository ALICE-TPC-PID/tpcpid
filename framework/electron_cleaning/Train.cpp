#include <cstdlib>
#include <iostream>
#include <string>

#include "TChain.h"
#include "TFile.h"
#include "TROOT.h"
#include "TString.h"
#include "TSystem.h"
#include "TTree.h"

#include "TMVA/DataLoader.h"
#include "TMVA/Factory.h"
#include "TMVA/TMVAGui.h"
#include "TMVA/Tools.h"

TTree *OpenChain(TFile *f, TString name_tree) {
  TChain *chain = new TChain(name_tree);
  TString name_file = f->GetName();

  // in the root directory, search for name_tree and if there's one, add it
  TTree *tree = (TTree *)f->Get(name_tree);
  if (tree) {
    chain->Add(name_file + TString("/") + TString(name_tree));
  }

  TList *list = f->GetListOfKeys();
  for (int i = 0; i < list->GetSize(); i++) {
    TKey *key = (TKey *)list->At(i);
    if (strcmp(key->GetClassName(), "TDirectoryFile") == 0) {
      if (string(key->GetName()).find("DF_") != string::npos)
        chain->Add(name_file + "/" + TString(key->GetName()) + TString("/") +
                   TString(name_tree));
    }
  }
  TTree* chain2 = chain->CloneTree();
  return chain2;
}

int Train(std::string input_file_path, std::string output_file_path, std::string weights_file_path) {
  TMVA::Tools::Instance();

  std::cout << std::endl;
  std::cout << "==> Start TMVAClassification (BDT only)" << std::endl;

  // --- Input data
  TFile* input = new TFile(input_file_path.c_str());
  if (!input || input->IsZombie()) {
    std::cerr << "ERROR: could not open data file" << std::endl;
    return 1;
  }

  TString name_tree = "O2tpcskimv0wde";
  // --- Output file
  TString outfilePath(output_file_path.c_str());
  std::unique_ptr<TFile> outputFile{TFile::Open(outfilePath, "RECREATE")};
  TTree *signalTree = OpenChain(input, name_tree);
  if (!outputFile || outputFile->IsZombie()) {
    std::cerr << "ERROR: could not open output file" << std::endl;
    return 1;
  }

  // --- Factory and DataLoader
  auto factory = std::make_unique<TMVA::Factory>(
      "TMVAClassification", outputFile.get(),
      "!V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;G,D:AnalysisType="
      "Classification");

  auto dataloader = std::make_unique<TMVA::DataLoader>("bdt");

  // --- Input variables
  dataloader->AddVariable("fGammaPsiPair", 'F');
  dataloader->AddVariable("fAlphaV0", 'F');
  dataloader->AddVariable("fCosPAV0", 'F');

  // --- Spectators (optional)
    dataloader->AddSpectator("fTPCInnerMom := 1./fTPCInnerParam", 'F');
  dataloader->AddSpectator("fNSigTPC", 'F');
  dataloader->AddSpectator("fTPCSignal", 'F');

  // --- Add trees
  dataloader->AddSignalTree(signalTree, 1.0);
  dataloader->AddBackgroundTree(signalTree, 1.0);
  // dataloader->SetBackgroundWeightExpression("weight");

  TCut signalCut =
      "fNSigTPC < 3 && fPidIndex == 0 && fNSigTPC > -3";
  TCut backgroundCut =
      "fPidIndex == 0 && (fNSigTPC < -3 || fNSigTPC > 3)";

  // --- Train/test split
  dataloader->PrepareTrainingAndTestTree(
      signalCut, backgroundCut,
      "nTrain_Signal=25000:nTrain_Background=2500:SplitMode=Random:NormMode="
      "NumEvents:!V");

  // --- Book BDT method (Adaptive Boost)
  factory->BookMethod(
      dataloader.get(), TMVA::Types::kBDT, "BDT",
      "!H:!V:NTrees=850:MinNodeSize=2.5%:MaxDepth=3:BoostType=AdaBoost:"
      "AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:"
      "SeparationType=GiniIndex:nCuts=20");

  // --- Train, test, evaluate
  factory->TrainAllMethods();
  factory->TestAllMethods();
  factory->EvaluateAllMethods();

  outputFile->Write();
  std::cout << "==> Wrote root file: " << outputFile->GetName() << std::endl;
  std::cout << "==> TMVAClassification is done!" << std::endl;

  if (!gROOT->IsBatch())
    TMVA::TMVAGui(outfilePath);

  return 0;
}
