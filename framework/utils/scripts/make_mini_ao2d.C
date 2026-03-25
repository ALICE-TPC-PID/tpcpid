#include <TFile.h>
#include <TTree.h>
#include <TKey.h>
#include <TDirectory.h>
#include <TClass.h>
#include <TSystem.h>

#include <iostream>
#include <vector>
#include <string>
#include <set>
#include <algorithm>
#include <cmath>

// example execution:
// root -l -b -q 'make_mini_ao2d.C("/lustre/alice/users/marin/R3A/TPCTreeswithNN/PbPbapass5_HR_251006/TPCTrees_HR_LHC23zzf_251006/AO2D_merge_LHC23zzf.root","AO2D_mini.root","[\"O2tpcskimv0wde\",\"O2tpctofskimwde\"]",10.0)'

struct TreeInfo {
  std::string path;
  std::string dir;
  std::string name;
};

std::string trim(const std::string& s) {
  const char* ws = " \t\n\r\"";
  size_t b = s.find_first_not_of(ws);
  if (b == std::string::npos) return "";
  size_t e = s.find_last_not_of(ws);
  return s.substr(b, e - b + 1);
}

std::vector<std::string> parsePatternList(const std::string& input) {
  std::vector<std::string> out;
  std::string s = input;

  s.erase(std::remove(s.begin(), s.end(), '['), s.end());
  s.erase(std::remove(s.begin(), s.end(), ']'), s.end());

  size_t start = 0;
  while (start < s.size()) {
    size_t comma = s.find(',', start);
    std::string token = (comma == std::string::npos) ? s.substr(start) : s.substr(start, comma - start);
    token = trim(token);
    if (!token.empty()) out.push_back(token);
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  return out;
}

void collectTreesRecursive(TDirectory* dir, const std::string& basePath, std::vector<TreeInfo>& trees) {
  TIter next(dir->GetListOfKeys());
  TKey* key = nullptr;

  while ((key = (TKey*)next())) {
    std::string keyName = key->GetName();
    std::string className = key->GetClassName();
    std::string fullPath = basePath.empty() ? keyName : basePath + "/" + keyName;

    TClass* cl = TClass::GetClass(className.c_str());
    if (!cl) continue;

    if (cl->InheritsFrom(TTree::Class())) {
      trees.push_back({fullPath, basePath, keyName});
    } else if (cl->InheritsFrom(TDirectory::Class())) {
      TDirectory* subdir = (TDirectory*)key->ReadObj();
      collectTreesRecursive(subdir, fullPath, trees);
    }
  }
}

TDirectory* mkdirRecursive(TFile* fout, const std::string& dirPath) {
  if (dirPath.empty()) return fout;

  TDirectory* current = fout;
  size_t start = 0;
  while (start < dirPath.size()) {
    size_t slash = dirPath.find('/', start);
    std::string part = (slash == std::string::npos) ? dirPath.substr(start)
                                                    : dirPath.substr(start, slash - start);
    if (!part.empty()) {
      TDirectory* next = dynamic_cast<TDirectory*>(current->Get(part.c_str()));
      if (!next) next = current->mkdir(part.c_str());
      current = next;
    }
    if (slash == std::string::npos) break;
    start = slash + 1;
  }
  return current;
}

void make_mini_ao2d(
    const char* inputFile = "/lustre/alice/users/marin/R3A/TPCTreeswithNN/PbPbapass5_HR_251006/TPCTrees_HR_LHC23zzf_251006/AO2D_merge_LHC23zzf.root",
    const char* outputFile = "AO2D_mini.root",
    const char* patterns = "[\"O2tpcskimv0wde\",\"O2tpctofskimwde\"]",
    double targetSizeMB = 10.0
) {
  std::cout << "Opening input file: " << inputFile << std::endl;
  TFile* fin = TFile::Open(inputFile, "READ");
  if (!fin || fin->IsZombie()) {
    std::cerr << "Error: could not open input file: " << inputFile << std::endl;
    return;
  }

  std::vector<TreeInfo> allTrees;
  collectTreesRecursive(fin, "", allTrees);

  std::cout << "\n=== All TTrees found in file ===" << std::endl;
  if (allTrees.empty()) {
    std::cout << "No trees found." << std::endl;
    fin->Close();
    return;
  }
  for (const auto& t : allTrees) {
    std::cout << "  " << t.path << std::endl;
  }

  std::vector<std::string> wanted = parsePatternList(patterns);

  std::cout << "\n=== Requested patterns ===" << std::endl;
  for (const auto& p : wanted) {
    std::cout << "  " << p << std::endl;
  }

  std::vector<TreeInfo> selected;
  std::set<std::string> usedPaths;

  for (const auto& pat : wanted) {
    bool found = false;
    for (const auto& t : allTrees) {
      if (t.name.find(pat) != std::string::npos) {
        if (usedPaths.insert(t.path).second) {
          selected.push_back(t);
          std::cout << "Matched pattern \"" << pat << "\" -> " << t.path << std::endl;
        }
        found = true;
        break;
      }
    }
    if (!found) {
      std::cout << "Warning: no tree found containing substring \"" << pat << "\"" << std::endl;
    }
  }

  if (selected.empty()) {
    std::cerr << "Error: no trees matched the requested patterns." << std::endl;
    fin->Close();
    return;
  }

  const double targetBytes = targetSizeMB * 1024.0 * 1024.0;
  const double targetBytesPerTree = targetBytes / static_cast<double>(selected.size());

  std::cout << "\nTarget output size: " << targetSizeMB << " MB" << std::endl;
  std::cout << "Selected trees: " << selected.size() << std::endl;
  std::cout << "Approximate target per selected tree: "
            << targetBytesPerTree / (1024.0 * 1024.0) << " MB" << std::endl;

  std::cout << "\nCreating output file: " << outputFile << std::endl;
  TFile* fout = TFile::Open(outputFile, "RECREATE");
  if (!fout || fout->IsZombie()) {
    std::cerr << "Error: could not create output file: " << outputFile << std::endl;
    fin->Close();
    return;
  }

  std::cout << "\n=== Copying entries ===" << std::endl;

  for (const auto& t : selected) {
    TTree* tin = dynamic_cast<TTree*>(fin->Get(t.path.c_str()));
    if (!tin) {
      std::cout << "Skipping unreadable tree: " << t.path << std::endl;
      continue;
    }

    Long64_t nEntries = tin->GetEntries();
    Long64_t zipBytes = tin->GetZipBytes();
    Long64_t totBytes = tin->GetTotBytes();

    double bytesPerEntry = 0.0;
    if (nEntries > 0 && zipBytes > 0) {
      bytesPerEntry = static_cast<double>(zipBytes) / static_cast<double>(nEntries);
    } else if (nEntries > 0 && totBytes > 0) {
      bytesPerEntry = static_cast<double>(totBytes) / static_cast<double>(nEntries);
    } else {
      bytesPerEntry = 1024.0; // conservative fallback
    }

    Long64_t entriesToCopy = static_cast<Long64_t>(std::floor(targetBytesPerTree / bytesPerEntry));
    if (entriesToCopy < 1) entriesToCopy = 1;
    if (entriesToCopy > nEntries) entriesToCopy = nEntries;

    std::cout << "Tree: " << t.path << std::endl;
    std::cout << "  entries          = " << nEntries << std::endl;
    std::cout << "  compressed bytes = " << zipBytes << std::endl;
    std::cout << "  total bytes      = " << totBytes << std::endl;
    std::cout << "  bytes/entry est. = " << bytesPerEntry << std::endl;
    std::cout << "  entries to copy  = " << entriesToCopy << std::endl;
    std::cout << "  estimated size   = "
              << (entriesToCopy * bytesPerEntry) / (1024.0 * 1024.0) << " MB" << std::endl;

    TDirectory* outDir = mkdirRecursive(fout, t.dir);
    outDir->cd();

    TTree* tout = tin->CloneTree(0);

    for (Long64_t i = 0; i < entriesToCopy; ++i) {
      tin->GetEntry(i);
      tout->Fill();
    }

    tout->Write();
    fout->cd();
  }

  fout->Write();
  fout->Close();
  fin->Close();

  TFile* fcheck = TFile::Open(outputFile, "READ");
  if (fcheck && !fcheck->IsZombie()) {
    Long64_t finalSizeBytes = fcheck->GetSize();
    std::cout << "\nDone. Wrote mini file: " << outputFile << std::endl;
    std::cout << "Final file size on disk: "
              << finalSizeBytes / (1024.0 * 1024.0) << " MB" << std::endl;
    fcheck->Close();
  } else {
    std::cout << "\nDone. Wrote mini file: " << outputFile << std::endl;
  }
}