#include "MRootGraphic.h"
#include "MRootIO.h"
#include "TGraph.h"
#include "TLegend.h"

#include "MRootGraphic.h"
#include "MRootIO.h"
#include "TGraph.h"
#include "TLegend.h"

TString gTag_nn = "NN";
TString gTag_bb = "BB";

void PlottingDetailed(TFile *file_nn, TFile *file_bb, TString path_output_graph,
                      TString tag_ncls) {
  TString name_tag = tag_ncls;
  TCanvas *c = new TCanvas("canvas_detailed_" + name_tag,
                           "canvas_detailed_" + name_tag, 1200, 800);

#define GetProfilePlotlottingDetailed(nn_bb, mean_sigma, var)                  \
  TProfile *var##_pion_##nn_bb = (TProfile *)file_##nn_bb->Get(                \
      "fRunNumber_" #var "_" + tag_ncls + "_Pion");                            \
  TProfile *var##_elec_##nn_bb = (TProfile *)file_##nn_bb->Get(                \
      "fRunNumber_" #var "_" + tag_ncls + "_Electron");

  GetProfilePlotlottingDetailed(nn, mean, dEdx);
  GetProfilePlotlottingDetailed(nn, mean, dEdx_exp);
  GetProfilePlotlottingDetailed(nn, sigma, delta_dEdx);
  GetProfilePlotlottingDetailed(nn, mean, fNSigTPC);

  GetProfilePlotlottingDetailed(bb, mean, dEdx);
  GetProfilePlotlottingDetailed(bb, mean, dEdx_exp);
  GetProfilePlotlottingDetailed(bb, sigma, delta_dEdx);
  GetProfilePlotlottingDetailed(bb, mean, fNSigTPC);

  c->Divide(4, 2);
  TProfile *p_nn, *p_bb;
  double maxy, miny;
  double maxy_user, miny_user;

#define SingleProfile(nn_bb, var, species) var##_##species##_##nn_bb

#define GroupProfile(...)                                                      \
  p_nn = SingleProfile(nn, __VA_ARGS__);                                       \
  p_bb = SingleProfile(bb, __VA_ARGS__);                                       \
  MRootGraphic::StyleHistCommon(p_nn);                                         \
  MRootGraphic::StyleHistCommon(p_bb);                                         \
  maxy = TMath::Max(p_nn->GetMaximum(), p_bb->GetMaximum());                   \
  miny = TMath::Min(p_nn->GetMinimum(), p_bb->GetMinimum());                   \
  maxy_user = maxy + 0.1 * (maxy - miny);                                      \
  miny_user = miny - 0.1 * (maxy - miny);                                      \
  p_nn->GetYaxis()->SetRangeUser(miny_user, maxy_user);                        \
  p_nn->SetLineColor(kRed);                                                    \
  p_bb->SetLineColor(kBlue);                                                   \
  p_nn->Draw();                                                                \
  p_bb->Draw("same");

  c->cd(1);
  GroupProfile(dEdx, elec);
  c->cd(2);
  GroupProfile(dEdx_exp, elec);
  c->cd(3);
  GroupProfile(delta_dEdx, elec);
  c->cd(4);
  GroupProfile(fNSigTPC, elec);
  c->cd(5);
  GroupProfile(dEdx, pion);
  c->cd(6);
  GroupProfile(dEdx_exp, pion);
  c->cd(7);
  GroupProfile(delta_dEdx, pion);
  c->cd(8);
  GroupProfile(fNSigTPC, pion);

  TLegend *leg = new TLegend(0.6, 0.7, 0.9, 0.9);
  leg->SetFillStyle(0);
  leg->SetLineStyle(0);
  leg->SetTextSize(0.04);
  leg->SetLineColor(0);
  leg->AddEntry(p_nn, gTag_nn, "L");
  leg->AddEntry(p_bb, gTag_bb, "L");
  leg->Draw("same");

  c->SaveAs(path_output_graph + "_detailed_" + name_tag + ".json");
}

void PlottingDetailed(TFile *file_nn, TFile *file_bb,
                      TString path_output_graph) {
  vector<TString> vec_tag_ncls = {"highNcls", "lowNcls", "AllNcls"};

  for (const auto &tag_ncls : vec_tag_ncls)
    PlottingDetailed(file_nn, file_bb, path_output_graph, tag_ncls);
}

void RunDependentProfile(TString path_input_nn = "~/test/sepPower_nn.root",
                         TString path_input_bb = "~/test/sepPower_bb.root",
                         TString path_output_graph = "~/test/sepPower_graph",
                         TString tag_nn = "NN", TString tag_bb = "BB") {
  TFile *file_nn = new TFile(path_input_nn, "READ");
  TFile *file_bb = new TFile(path_input_bb, "READ");

  gTag_bb = tag_bb;
  gTag_nn = tag_nn;

  MRootGraphic::StyleCommon();
  gStyle->SetPadTickX(1);
  gStyle->SetPadTickY(1);
  gStyle->SetPadGridX(1);
  gStyle->SetPadGridY(1);
  PlottingDetailed(file_nn, file_bb, path_output_graph);
  // PlottingSepPower(file_nn, file_bb, path_output_graph);
}

int main(int argc, char **argv) {
  TString path_input_nn = "~/test/sepPower_nn.root";
  TString path_input_bb = "~/test/sepPower_bb.root";
  TString path_output_graph = "~/test/sepPower_graph";
  TString tag_nn = "NN";
  TString tag_bb = "BB";

  if (argc > 1)
    path_input_nn = argv[1];
  if (argc > 2)
    path_input_bb = argv[2];
  if (argc > 3)
    path_output_graph = argv[3];
  if (argc > 4)
    tag_nn = argv[4];
  if (argc > 5)
    tag_bb = argv[5];

  RunDependentProfile(path_input_nn, path_input_bb, path_output_graph, tag_nn,
                      tag_bb);
  return 0;
}
