#include "MHead.h"
#include "MRootGraphic.h"
#include "MRootIO.h"
#include "TApplication.h"
#include "TColor.h"
#include "TLegend.h"
#include "TProfile.h"
#include "TStyle.h"

TFile *gFileInput = nullptr;
TFile *gFileOutput = nullptr;

typedef struct StrGausFit {
  TH1D *hMean;
  TH1D *hSigma;
} StrGausFit;

StrGausFit AddFit(TH2 *h2d, double limit_low, double limit_high,
                  bool no_fit = true) {
  // TF1 *f1 = new TF1("f1", "gaus"); change it to double
  TString name_h2d = h2d->GetName();
  TString title_h2d = h2d->GetTitle();
  TString titleY_h2d = h2d->GetYaxis()->GetTitle();

  TF1 *f1 = new TF1("f1", "gaus");
  f1->SetRange(limit_low, limit_high);
  TObjArray aSlices_electron;

  h2d->FitSlicesY(f1, 0, -1, 0, "QNR", &aSlices_electron);
  // aSlices_electron.SetOwner(0);

  //--- Three objects: Mean, Sigma, Chi2
  TH1D *hMean = (TH1D *)aSlices_electron.At(1)->Clone(name_h2d + "_mean");
  hMean->SetTitle(title_h2d);
  hMean->GetYaxis()->SetTitle("mean " + titleY_h2d);
  TH1D *hSigma = (TH1D *)aSlices_electron.At(2)->Clone(name_h2d + "_sigma");
  hSigma->SetTitle(title_h2d);
  hSigma->GetYaxis()->SetTitle("sigma " + titleY_h2d);

  if (no_fit) {
    // set hMean and hSigma as profile histograms
    for (int i_x_2d = 1; i_x_2d <= h2d->GetNbinsX(); i_x_2d++) {
      TH1D *h1_temp = (TH1D *)h2d->ProjectionY("h1_temp", i_x_2d, i_x_2d);
      double mean = h1_temp->GetMean();
      double sigma = h1_temp->GetRMS();
      hMean->SetBinContent(i_x_2d, mean);
      hMean->SetBinError(i_x_2d, 0);
    }
  }

  gFileOutput->cd();
  h2d->Write();
  hMean->Write();
  hSigma->Write();

  delete f1;
  return {hMean, hSigma};
}

void CalculateSeparation(StrGausFit fit_dedx_electron, StrGausFit fit_dedx_Pion,
                         StrGausFit fit_DeltaDeDx_electron,
                         StrGausFit fit_DeltaDedx_Pion,
                         const char *name_result) {
  TH1D *h_result = (TH1D *)fit_dedx_electron.hMean->Clone(name_result);
  h_result->Add(fit_dedx_Pion.hMean, -1);
  h_result->Add(fit_DeltaDeDx_electron.hMean, -1);
  h_result->Add(fit_DeltaDedx_Pion.hMean, 1);
  h_result->SetTitle(h_result->GetTitle());
  h_result->GetYaxis()->SetTitle("Separation Power");

  for (int ibin = 1; ibin <= h_result->GetNbinsX(); ibin++) {
    double sigma1 = fit_DeltaDeDx_electron.hSigma->GetBinContent(ibin);
    double sigma2 = fit_DeltaDedx_Pion.hSigma->GetBinContent(ibin);
    double sigma = sigma1 + sigma2;
    double value = h_result->GetBinContent(ibin) / sigma * 2.0;
    h_result->SetBinContent(ibin, value);
    h_result->SetBinError(ibin, 0);
  }
  h_result->Write();
}

enum NN_BB { kNN = 0, kBB = 1 };

void GetSeparationPower(TString tag_x = "fTgl", TString tag_ncls = "AllNcls",
                        int nn_bb = kNN) {
  TH2D *dEdx_elec, *dEdx_pion;
  switch (nn_bb) {
  case 0:
    dEdx_elec =
        (TH2D *)gFileInput->Get(tag_x + "_dEdx_exp_" + tag_ncls + "_Electron");
    dEdx_pion =
        (TH2D *)gFileInput->Get(tag_x + "_dEdx_exp_" + tag_ncls + "_Pion");
    break;
  case 1:
    dEdx_elec =
        (TH2D *)gFileInput->Get(tag_x + "_dEdx_" + tag_ncls + "_Electron");
    dEdx_pion = (TH2D *)gFileInput->Get(tag_x + "_dEdx_" + tag_ncls + "_Pion");
  default:
    break;
  }
  dEdx_elec->SetName(tag_x + "_dEdx_" + tag_ncls + "_Electron");
  dEdx_pion->SetName(tag_x + "_dEdx_" + tag_ncls + "_Pion");
  dEdx_elec->GetYaxis()->SetTitle("dEdx Electron");
  dEdx_pion->GetYaxis()->SetTitle("dEdx Pion");

  if (!dEdx_elec || !dEdx_pion) {
    cerr << "Error: GetSeparationPower: dEdx_elec or dEdx_pion is null" << endl;
    return;
  }

  TH2D *delta_dEdx_elec =
      (TH2D *)gFileInput->Get(tag_x + "_delta_dEdx_" + tag_ncls + "_Electron");
  TH2D *delta_dEdx_pion =
      (TH2D *)gFileInput->Get(tag_x + "_delta_dEdx_" + tag_ncls + "_Pion");

  StrGausFit fit_dEdx_elec = AddFit(dEdx_elec, 60, 100, nn_bb == kNN);
  StrGausFit fit_dEdx_pion = AddFit(dEdx_pion, 40, 60, nn_bb == kNN);
  StrGausFit fit_delta_dEdx_elec = AddFit(delta_dEdx_elec, -20, 20);
  StrGausFit fit_delta_dEdx_pion = AddFit(delta_dEdx_pion, -20, 20);

  CalculateSeparation(fit_dEdx_elec, fit_dEdx_pion, fit_delta_dEdx_elec,
                      fit_delta_dEdx_pion, tag_x + "_sepPower_" + tag_ncls);
}

void CalculateSeparationPower(TString path_input = "~/test/output.root",
                              TString path_output = "~/test/sepPower.root",
                              int nn_bb = kNN) {
  gFileInput = new TFile(path_input, "READ");
  gFileOutput = new TFile(path_output, "RECREATE");

  vector<TString> vec_tag_x = {"fTgl", "fFt0Occ"};
  vector<TString> vec_tag_ncls = {"highNcls", "lowNcls", "AllNcls"};

  for (const auto &tag_x : vec_tag_x) {
    for (const auto &tag_ncls : vec_tag_ncls) {
      GetSeparationPower(tag_x, tag_ncls, nn_bb);
    }
  }

  gFileOutput->Close();
  gFileInput->Close();
  delete gFileOutput;
  delete gFileInput;
}

void SeparationPower(TString path_input = "~/test/output.root",
                     TString path_output = "~/test/sepPower.root",
                     int nn_bb = kNN) {
  CalculateSeparationPower(path_input, path_output, nn_bb);
}

int main(int argc, char **argv) {
  TString path_input = "~/test/output.root";
  TString path_output = "~/test/sepPower.root";
  int nn_bb = kNN;

  if (argc > 1)
    path_input = argv[1];
  if (argc > 2)
    path_output = argv[2];
  if (argc > 3)
    nn_bb = atoi(argv[3]);

  SeparationPower(path_input, path_output, nn_bb);

  return 0;
}