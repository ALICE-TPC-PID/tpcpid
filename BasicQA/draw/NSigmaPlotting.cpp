#include "MHist.h"
#include "MRootGraphic.h"
#include "MRootIO.h"
#include "TGraph.h"
#include "TLegend.h"


void NSigmaPlotting(
    TString path_input = "/home/szhu/test/SkimmingTreeRead.root",
    TString path_output = "./testPlot") {
  TFile *file_input = new TFile(path_input, "READ");
  TFile *file_output = new TFile(path_output + ".root", "RECREATE");

  vector<array<string, 2>> conditions2_type = {{"isElectron", "Electron"},
                                               {"isPion", "Pion"}};
  vector<array<string, 2>> conditions_cluster = {
      {"nCluster > 130", "highNcls"},
      {"nCluster < 130 && nCluster >80", "lowNcls"},
      {"nCluster > 80", "AllNcls"},
  };

  str_cond str_cond_type(conditions2_type);
  str_cond str_cond_clus(conditions_cluster);
  using MIndexCond = MIndexAny<str_cond>;
  MIndexCond index_type(str_cond_type);
  // MIndexCond index_clus(str_cond);

  // using MIndexCond = MIndexAny<str_cond>;

  vector<vector<MHnTool *>> vec_vec_hnt;

  for (auto i_type : index_type) {
    cout << (int)i_type << endl;
  }

  // MVec<MHnTool

  for (auto cond_cluster : conditions_cluster) {
    vector<MHnTool *> vec_hnt;
    for (auto cond_type : conditions2_type) {
      TString name = TString("pIn_fFt0Occ_fNSigTPC_fTgl_") +
                     TString(cond_cluster[1]) + TString("_") +
                     TString(cond_type[1]);
      THnD *h = (THnD *)file_input->Get(name);
      MHnTool *hnt = new MHnTool(h);
      if (!h) {
        cerr << "Error: NSigmaPlotting: " << name << " not found in file_input"
             << endl;
        continue;
      }
      //////////////////////////// rebin ////////////////////////////
      hnt->Rebin(0, 10); // Rebin pIn
      hnt->Rebin(1, 2);  // Rebin fFt0Occ
      hnt->Rebin(3, 2);  // Rebin Tgl
      ///////////////////////////////////////////////////////////////
      vec_hnt.push_back(hnt);
    }
    vec_vec_hnt.push_back(vec_hnt);
  }

  vec_vec_hnt[0][0]->PrintAllAxis();

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

  // var_pIn.rebin(10);
  // var_fFt0Occ.rebin(2);
  MIndexHist index_pIn(var_pIn, 1, 10);
  MIndexHist index_fFt0Occ(var_fFt0Occ, 1, 2);
  MIndexHist index_fNSigTPC(var_fNSigTPC);
  MIndexHist index_fTgl(var_fTgl, 1, 10);

  MHist1D h1_pIn_Sigma(index_pIn, "Sigma");
  MHist1D h1_pIn_Mean(index_pIn, "Mean");

  // MVec<MHist1D> vec_fTgl_h2_pIn_nSigma(index_fTgl, h1_pIn_Sigma);
  // MVec<MVec<MHist1D>> vec_fNSigTPC_vec_fTgl_h2_pIn_nSigma(
  //     index_fNSigTPC, vec_fTgl_h2_pIn_nSigma);
  vector<vector<MVec<MVec<MHist1D>>>> vec_all_sigma;
  vector<vector<MVec<MVec<MHist1D>>>> vec_all_mean;

  cout << "Starting to create histograms..." << endl;
  for (auto cond_cluster : conditions_cluster) {
    vector<MVec<MVec<MHist1D>>> vec_fNSigTPC_vec_fTgl_h2_pIn_sigma;
    vector<MVec<MVec<MHist1D>>> vec_fNSigTPC_vec_fTgl_h2_pIn_mean;
    for (auto cond_type : conditions2_type) {
      TString name = cond_cluster[1] + "_" + cond_type[1];
      TString title_sigma = "Width " + cond_type[1] + " " + cond_cluster[0];
      TString title_mean = "Mean " + cond_type[1] + " " + cond_cluster[0];
      gDirectory = nullptr;
      MHist1D h1_pIn_Sigma(index_pIn, "Sigma_" + name, title_sigma);
      MHist1D h1_pIn_Mean(index_pIn, "Mean_" + name, title_mean);
      h1_pIn_Mean.fHisto->SetDirectory(0);
      h1_pIn_Sigma.fHisto->SetDirectory(0);
      // MHist1D h1_pIn_Mean(index_pIn, "Mean_" + name);
      MVec<MHist1D> vec_fTgl_h2_pIn_nSigma(index_fTgl, h1_pIn_Sigma);
      MVec<MVec<MHist1D>> vec_temp(index_fFt0Occ, vec_fTgl_h2_pIn_nSigma);
      MVec<MHist1D> vec_fTgl_h2_pIn_nMean(index_fTgl, h1_pIn_Mean);
      MVec<MVec<MHist1D>> vec_temp_mean(index_fFt0Occ, vec_fTgl_h2_pIn_nMean);
      file_output->cd();
      vec_fNSigTPC_vec_fTgl_h2_pIn_sigma.push_back(vec_temp);
      vec_fNSigTPC_vec_fTgl_h2_pIn_mean.push_back(vec_temp_mean);
      static int count = 0;
      cout << "\r" << count++ << flush;
    }
    vec_all_mean.push_back(vec_fNSigTPC_vec_fTgl_h2_pIn_mean);
    vec_all_sigma.push_back(vec_fNSigTPC_vec_fTgl_h2_pIn_sigma);
  }

  cout << "Histograms created." << endl;
  // Loop over all histograms and fit them
  cout << "Starting to fit histograms..." << endl;
  for (auto i_fFt0Occ : index_fFt0Occ) {
    for (auto i_fTgl : index_fTgl) {
      for (auto i_pIn : index_pIn) {
        for (int i_clus = 0; i_clus < conditions_cluster.size(); i_clus++) {
          for (int i_type = 0; i_type < conditions2_type.size(); i_type++) {
            auto &hnt = vec_vec_hnt[i_clus][i_type];
            auto h_nsigma = hnt->Project(2, {i_pIn, i_fFt0Occ, i_fTgl});
            TF1 f_gauss("f_gauss", "gaus", -5, 5);
            h_nsigma->Fit(&f_gauss, "Q");
            MDouble sigma(f_gauss.GetParameter(2), f_gauss.GetParError(2));
            MDouble mean(f_gauss.GetParameter(1), f_gauss.GetParError(1));

            vec_all_sigma[i_clus][i_type].currentObject().SetBinInfo(sigma);
            vec_all_mean[i_clus][i_type].currentObject().SetBinInfo(mean);
          }
        }
      }
    }
  }
  cout << "Fitting completed." << endl;

  file_output->cd();
  for (auto &i : vec_all_sigma)
    for (auto &j : i)
      j.Write();

  for (auto &i : vec_all_mean)
    for (auto &j : i)
      j.Write();

  file_output->Close();
}