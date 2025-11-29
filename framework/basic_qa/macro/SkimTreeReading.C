#define MRDF
#include "MHead.h"
#include "MHist.h"
#include "MRootIO.h"
#include <ROOT/RDataFrame.hxx>
#include <set>

void SkimTreeReading(TString path_input = "../config/SkimTreeReading.root",
                     TString path_ouptut = "output.root",
                     TString name_tree = "O2tpcskimv0tree") {
  ROOT::EnableImplicitMT();
  TChain *chain = MRootIO::OpenChain(path_input, name_tree);

  RDataFrame rdf_pre(*chain);

  auto rdf =
      rdf_pre.Alias("pIn", "fTPCInnerParam")
          .Alias("dEdx", "fTPCSignal")
          .Define("nCluster",
                  [](float fNormNClustersTPC) {
                    return 152 / (fNormNClustersTPC * fNormNClustersTPC);
                  },
                  {"fNormNClustersTPC"})
          .Define("dEdx_exp",
                  [](float fInvDeDxExpTPC) { return 1.f / fInvDeDxExpTPC; },
                  {"fInvDeDxExpTPC"})
          .Define("delta_dEdx",
                  [](float dEdx, float dEdx_exp) { return dEdx - dEdx_exp; },
                  {"dEdx", "dEdx_exp"})
          .Define(
              "IsAtFermiPlatu",
              [](float pIn) { return pIn < 3.5 * 0.139 && pIn > 3. * 0.139; },
              {"pIn"})
          .Define("isElectron",
                  [](UChar_t fPidIndex) { return fPidIndex == 0; },
                  {"fPidIndex"})
          .Define("isPion", [](UChar_t fPidIndex) { return fPidIndex == 2; },
                  {"fPidIndex"})
          /* .Define("isKaon", [](UChar_t fPidIndex) { return fPidIndex == 3; },
                  {"fPidIndex"}) */
          .Define("isProton", [](UChar_t fPidIndex) { return fPidIndex == 4; },
                  {"fPidIndex"});
  // add processing bar
  ROOT::RDF::Experimental::AddProgressBar(rdf);
  //     double axisOccuFt0[] = {0.,     130.,   1010.,  2740.,  5130., 8070.,
  //                           11590., 16010., 22030., 31840., 5.e4};
  // #define axisOccuFt0 10, axisOccuFt0
  // #define axisTgl 10, -1, 1
  // #define axis_dEdx 150, 10, 160
  // #define axis_fNSigTPC 100, -5, 5
  // #define axis_DeltaDeDx 160, -40, 40

  StrVar4Hist var_fFt0Occ("fFt0Occ", "Occupancy FT0C", "", 10,
                          {0., 130., 1010., 2740., 5130., 8070., 11590., 16010.,
                           22030., 31840., 5.e4});
  StrVar4Hist var_fTgl("fTgl", "Tgl", "", 10, {-1, 1});
  StrVar4Hist var_pIn("pIn", "p_{in}", "GeV/c", 100, GetLogBin(100, 0.1, 10));

  StrVar4Hist var_dEdx("dEdx", "dE/dx", "", 150, {10, 160});
  StrVar4Hist var_fNSigTPC("fNSigTPC", "n#sigma_{TPC}", "", 100, {-5, 5});
  StrVar4Hist var_dEdx_exp("dEdx_exp", "dE/dx exp", "", 150, {10, 160});
  StrVar4Hist var_delta_dEdx("delta_dEdx", "dE/dx - dE/dx exp", "", 80,
                             {-40, 40});

  /* #region: histrograms for separation power calculation */
  vector<StrVar4Hist> vec_str_x = {var_fFt0Occ, var_fTgl};
  vector<StrVar4Hist> vec_str_y = {var_dEdx, var_dEdx_exp, var_delta_dEdx,
                                   var_fNSigTPC};

  TFile *fOutput = new TFile(path_ouptut, "RECREATE");

  vector<array<string, 2>> conditions1_sepPower = {
      {"nCluster > 130", "highNcls"},
      {"nCluster < 130 && nCluster >80", "lowNcls"},
      {"nCluster > 80", "AllNcls"},
  };

  vector<array<string, 2>> conditions2_sepPower = {{"isElectron", "Electron"},
                                                   {"isPion", "Pion"}};

#define obj2push_thnd(rdf2push, ...)                                           \
  TupleTHnDModel tuple_thnd = GetTHnDModelWithTitle(__VA_ARGS__);              \
  gRResultHandles.push_back(                                                   \
      rdf2push.HistoND(get<0>(tuple_thnd), get<1>(tuple_thnd)));

  auto rdf_mip = rdf.Filter("IsAtFermiPlatu");
  for (auto cond1 : conditions1_sepPower) {
    string condition1 = cond1[0];
    string tag1 = cond1[1];
    for (auto cond2 : conditions2_sepPower) {
      string condition2 = cond2[0];
      string tag2 = cond2[1];
      TString tag = tag1 + "_" + tag2;

      auto rdf_selected = rdf.Filter(condition1).Filter(condition2);
      // pIn_fFt0Occ_fNSigTPC_fTgl
      obj2push_thnd(rdf_selected,
                    {var_pIn, var_fFt0Occ, var_fNSigTPC, var_fTgl}, condition1,
                    tag);

      auto rdf_selected_mip = rdf_mip.Filter(condition1).Filter(condition2);
      // fFt0Occ,fTgl:dEdx,dEdx_exp,delta_dEdx,fNSigTPC
      for (auto str_x : vec_str_x) {
        for (auto str_y : vec_str_y) {
          TString title(condition1);
          TString title_x = str_x.fTitle;
          TString title_y = str_y.fTitle + " " + tag2;
          if (str_x.fUnit != "") {
            title_x += " (" + str_x.fUnit + ")";
          }
          if (str_y.fUnit != "") {
            title_y += " (" + str_y.fUnit + ")";
          }
          // h2->SetTitle(title + ";" + title_x + ";" + title_y);
          gRResultHandles.push_back(rdf_selected_mip.Histo2D(
              GetTH2DModelWithTitle(str_x, str_y,
                                    title + ";" + title_x + ";" + title_y, tag),
              str_x.fName, str_y.fName));
        }
      }
    }
  }
  /* #endregion */

  // get run number
  auto unique_runs_rp = rdf.Aggregate(
      [](std::set<int> &acc, int run) {
        acc.insert(run);
        return acc;
      },
      [](const std::set<int> &a, const std::set<int> &b) {
        std::set<int> result = a;
        result.insert(b.begin(), b.end());
        return result;
      },
      "fRunNumber", std::set<int>());

  gRResultHandles.push_back(unique_runs_rp);
  RunGraphs(gRResultHandles);

  set<int> unique_runs = unique_runs_rp.GetValue();
  for (const auto &run : unique_runs) {
    cout << "Unique run number: " << run << endl;
  }

  auto rdf_withRun =
      rdf.Define("index_runNumber",
                 [unique_runs](int i) {
                   for (int index = 0; index < unique_runs.size(); index++) {
                     auto it = std::next(unique_runs.begin(), index);
                     if (*it == i) {
                       return index;
                     }
                   }
                   return -1; // Return -1 if not found
                 },
                 {"fRunNumber"});

  vector<RResultHandle> gRResultHandles2;
  auto rdf_withRun_mip = rdf_withRun.Filter("IsAtFermiPlatu");
  for (auto cond1 : conditions1_sepPower) {
    string condition1 = cond1[0];
    string tag1 = cond1[1];
    for (auto cond2 : conditions2_sepPower) {
      string condition2 = cond2[0];
      string tag2 = cond2[1];

      auto rdf_selected = rdf_withRun_mip.Filter(condition1).Filter(condition2);

      for (auto str_y : vec_str_y) {
        TString tag = tag1 + "_" + tag2;
        TString name = "fRunNumber_" + str_y.fName + "_" + tag;
        TString title(condition1);
        TString title_x = "Run";
        TString title_y = "<" + str_y.fTitle + " " + tag2 + ">";
        if (str_y.fUnit != "") {
          title_y += " (" + str_y.fUnit + ")";
        }
        gRResultHandles2.push_back(rdf_selected.Profile1D(
            {name, title + ";" + title_x + ";" + title_y, unique_runs.size(),
             -0.5, unique_runs.size() - 0.5},
            "index_runNumber", str_y.fName));
      }
    }
  }

  ROOT::RDF::Experimental::AddProgressBar(rdf_withRun_mip);
  RunGraphs(gRResultHandles2);

  // set lable for profile_run
  for (auto profile : gRResultHandles2) {
    TProfile *p = profile.GetPtr<TProfile>();
    if (p) {
      for (int i = 0; i < unique_runs.size(); i++) {
        auto it = std::next(unique_runs.begin(), i);
        p->GetXaxis()->SetBinLabel(i + 1, Form("%d", *it));
      }
    }
  }

  fOutput->cd();
  RResultWrite(gRResultHandles);
  RResultWrite(gRResultHandles2);
  fOutput->Close();
}

int main(int argc, char **argv) {
  if (argc < 3) {
    cout << "Usage: " << argv[0] << " <input_file> <output_file>" << endl;
    return 1;
  }

  TString path_input = argv[1];
  TString path_output = argv[2];
  TString name_tree = "O2tpcskimv0tree";

  if (argc == 4)
    name_tree = argv[3];

  SkimTreeReading(path_input, path_output, name_tree);

  return 0;
}