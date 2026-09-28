#include <TClass.h>
#include <TDirectory.h>
#include <TFile.h>
#include <TKey.h>
#include <TObject.h>
#include <TSystem.h>
#include <TTree.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

// Make a small, O2Physics-readable AO2D fixture by copying complete DF_* data-frame
// directories. This intentionally does not truncate individual trees: AO2D tables
// contain cross-table indices, and per-tree entry slicing can produce invalid input.
//
// Example:
// root -l -b -q 'make_mini_ao2d_full.C("/path/to/AO2D.root","mini_AO2D_full.root",1)'

namespace
{
bool inheritsFrom(const char* className, const TClass* base)
{
  TClass* cl = TClass::GetClass(className);
  return cl && cl->InheritsFrom(base);
}

bool startsWith(const std::string& value, const std::string& prefix)
{
  return value.rfind(prefix, 0) == 0;
}

TDirectory* mkdirIn(TDirectory* parent, const char* name)
{
  TDirectory* existing = dynamic_cast<TDirectory*>(parent->Get(name));
  if (existing) {
    return existing;
  }
  return parent->mkdir(name);
}

void copyObject(TKey* key, TDirectory* outDir)
{
  TObject* obj = key->ReadObj();
  if (!obj) {
    std::cerr << "Warning: could not read object " << key->GetName() << std::endl;
    return;
  }

  outDir->cd();
  if (obj->InheritsFrom(TTree::Class())) {
    auto* tree = static_cast<TTree*>(obj);
    TTree* cloned = tree->CloneTree(-1, "fast");
    if (cloned) {
      cloned->Write(key->GetName(), TObject::kOverwrite);
    }
  } else {
    obj->Write(key->GetName(), TObject::kOverwrite);
  }
  delete obj;
}

void copyDirectoryRecursive(TDirectory* inDir, TDirectory* outDir)
{
  TIter next(inDir->GetListOfKeys());
  TKey* key = nullptr;

  while ((key = static_cast<TKey*>(next()))) {
    const std::string className = key->GetClassName();
    const std::string keyName = key->GetName();

    if (inheritsFrom(className.c_str(), TDirectory::Class())) {
      TDirectory* childIn = dynamic_cast<TDirectory*>(key->ReadObj());
      if (!childIn) {
        std::cerr << "Warning: could not read directory " << keyName << std::endl;
        continue;
      }
      TDirectory* childOut = mkdirIn(outDir, keyName.c_str());
      copyDirectoryRecursive(childIn, childOut);
      delete childIn;
      continue;
    }

    copyObject(key, outDir);
  }
}

std::vector<std::string> listDataFrameDirs(TFile* input)
{
  std::vector<std::string> dirs;
  TIter next(input->GetListOfKeys());
  TKey* key = nullptr;

  while ((key = static_cast<TKey*>(next()))) {
    const std::string keyName = key->GetName();
    if (!startsWith(keyName, "DF_")) {
      continue;
    }
    if (inheritsFrom(key->GetClassName(), TDirectory::Class())) {
      dirs.push_back(keyName);
    }
  }

  std::sort(dirs.begin(), dirs.end());
  dirs.erase(std::unique(dirs.begin(), dirs.end()), dirs.end());
  return dirs;
}
} // namespace

void make_mini_ao2d_full(
    const char* inputFile = "/lustre/alice/users/csonnab/cern-fellowship/run/softbombs/NN/data/Mesut100kIRpp13TeV-20260706-120347/001/AO2D.root",
    const char* outputFile = "/lustre/alice/users/csonnab/TPC/o2-tpc-pid/run/ci/data/mini_AO2D_full.root",
    int maxDataFrames = 1)
{
  std::cout << "Opening input AO2D: " << inputFile << std::endl;
  TFile* fin = TFile::Open(inputFile, "READ");
  if (!fin || fin->IsZombie()) {
    std::cerr << "Error: could not open input file: " << inputFile << std::endl;
    return;
  }

  std::vector<std::string> dfDirs = listDataFrameDirs(fin);
  if (dfDirs.empty()) {
    std::cerr << "Error: no top-level DF_* directories found. This is not a full AO2D-style file." << std::endl;
    fin->Close();
    return;
  }

  if (maxDataFrames <= 0 || maxDataFrames > static_cast<int>(dfDirs.size())) {
    maxDataFrames = static_cast<int>(dfDirs.size());
  }

  std::cout << "Found " << dfDirs.size() << " data-frame directories." << std::endl;
  std::cout << "Copying " << maxDataFrames << " complete data-frame director"
            << (maxDataFrames == 1 ? "y" : "ies") << "." << std::endl;

  gSystem->mkdir(gSystem->DirName(outputFile), true);
  TFile* fout = TFile::Open(outputFile, "RECREATE");
  if (!fout || fout->IsZombie()) {
    std::cerr << "Error: could not create output file: " << outputFile << std::endl;
    fin->Close();
    return;
  }
  fout->SetCompressionSettings(fin->GetCompressionSettings());

  for (int i = 0; i < maxDataFrames; ++i) {
    const std::string& dfName = dfDirs[i];
    std::cout << "Copying " << dfName << std::endl;
    TDirectory* inDir = dynamic_cast<TDirectory*>(fin->Get(dfName.c_str()));
    TDirectory* outDir = fout->mkdir(dfName.c_str());
    if (!inDir || !outDir) {
      std::cerr << "Warning: could not open/create " << dfName << std::endl;
      continue;
    }
    copyDirectoryRecursive(inDir, outDir);
  }

  fout->Close();
  fin->Close();

  TFile* fcheck = TFile::Open(outputFile, "READ");
  if (fcheck && !fcheck->IsZombie()) {
    std::cout << "Wrote " << outputFile << std::endl;
    std::cout << "Output size: " << fcheck->GetSize() / (1024.0 * 1024.0) << " MB" << std::endl;
    fcheck->Close();
  }
}
