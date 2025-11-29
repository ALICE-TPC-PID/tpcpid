#include "MRootGraphic.h"
#include "MRootIO.h"
#include "TGraph.h"
#include "TLegend.h"

TString gTag_nn = "NN";
TString gTag_bb = "BB";
bool gNoOccupancy = false;

TGraph *GetGraph(TH1D *h1, TString name) {
  TGraph *g = new TGraph(h1->GetNbinsX());
  for (int i = 0; i < h1->GetNbinsX(); i++) {
    g->SetPoint(i, h1->GetBinCenter(i + 1), h1->GetBinContent(i + 1));
  }
  g->SetName(name);
  g->SetTitle(h1->GetTitle());
  g->GetXaxis()->SetTitle(h1->GetXaxis()->GetTitle());
  g->GetYaxis()->SetTitle(h1->GetYaxis()->GetTitle());
  return g;
}

void PlottingDetailed(TFile *file_nn, TFile *file_bb, TString path_output_graph,
                      TString tag_x, TString tag_ncls) {
  TString name_tag = tag_x + "_" + tag_ncls;
  TCanvas *c = new TCanvas("canvas_detailed_" + name_tag,
                           "canvas_detailed_" + name_tag, 800, 800);

#define GetGraphPlotlottingDetailed(nn_bb, mean_sigma, var)                    \
  TH1D *mean_sigma##_##nn_bb##_##var##elec = (TH1D *)file_##nn_bb->Get(        \
      tag_x + "_" + #var + "_" + tag_ncls + "_Electron_" #mean_sigma);         \
  TGraph *g_##mean_sigma##_##nn_bb##_##var##elec = GetGraph(                   \
      mean_sigma##_##nn_bb##_##var##elec,                                      \
      "g_" + tag_x + "_" + #var + "_" + tag_ncls + "_" + #nn_bb + "_elec");    \
  TH1D *mean_sigma##_##nn_bb##_##var##pion = (TH1D *)file_##nn_bb->Get(        \
      tag_x + "_" + #var + "_" + tag_ncls + "_Pion_" #mean_sigma);             \
  TGraph *g_##mean_sigma##_##nn_bb##_##var##pion = GetGraph(                   \
      mean_sigma##_##nn_bb##_##var##pion,                                      \
      "g_" + tag_x + "_" + #var + "_" + tag_ncls + "_" + #nn_bb + "_pion");

  GetGraphPlotlottingDetailed(nn, mean, dEdx);
  GetGraphPlotlottingDetailed(nn, sigma, delta_dEdx);
  GetGraphPlotlottingDetailed(bb, mean, dEdx);
  GetGraphPlotlottingDetailed(bb, sigma, delta_dEdx);

  TGraph *g_nn, *g_bb;
  TH1D *h_frame;
  double maxy, miny;
  double maxy_user, miny_user;
  double maxx_user, minx_user;

  c->Divide(2, 2);
#define SingleGraph(nn_bb, mean_sigma, var, species)                           \
  g_##mean_sigma##_##nn_bb##_##var##species

#define SingleHist(n_bb, mean_sigma, var, species)                             \
  mean_sigma##_##n_bb##_##var##species

#define GroupGraph(...)                                                        \
  g_nn = SingleGraph(nn, __VA_ARGS__);                                         \
  g_bb = SingleGraph(bb, __VA_ARGS__);                                         \
  MRootGraphic::StyleHistCommon(g_nn);                                         \
  MRootGraphic::StyleHistCommon(g_bb);                                         \
  maxy = TMath::Max(SingleHist(nn, __VA_ARGS__)->GetMaximum(),                 \
                    SingleHist(bb, __VA_ARGS__)->GetMaximum());                \
  miny = TMath::Min(SingleHist(nn, __VA_ARGS__)->GetMinimum(),                 \
                    SingleHist(bb, __VA_ARGS__)->GetMinimum());                \
  maxx_user = g_nn->GetXaxis()->GetXmax();                                     \
  minx_user = g_nn->GetXaxis()->GetXmin();                                     \
  maxy_user = maxy + 0.1 * (maxy - miny);                                      \
  miny_user = miny - 0.1 * (maxy - miny);                                      \
  g_nn->SetLineColor(kRed);                                                    \
  g_bb->SetLineColor(kBlue);                                                   \
  h_frame = new TH1D(Form("h_frame_%d", GenerateUID()), "", 1, minx_user,      \
                     maxx_user);                                               \
  h_frame->GetYaxis()->SetRangeUser(miny_user, maxy_user);                     \
  h_frame->GetXaxis()->SetTitle(g_nn->GetXaxis()->GetTitle());                 \
  h_frame->GetYaxis()->SetTitle(g_nn->GetYaxis()->GetTitle());                 \
  h_frame->SetTitle(g_nn->GetTitle());                                         \
  MRootGraphic::StyleHistCommonHist(h_frame);                                  \
  h_frame->Draw("AXIS");                                                       \
  if (gTag_nn != "")                                                           \
    g_nn->Draw("L same");                                                      \
  if (gTag_bb != "")                                                           \
    g_bb->Draw("L same");

  c->cd(1);
  GroupGraph(mean, dEdx, elec);
  c->cd(2);
  GroupGraph(sigma, delta_dEdx, elec);
  c->cd(3);
  GroupGraph(mean, dEdx, pion);
  c->cd(4);
  GroupGraph(sigma, delta_dEdx, pion);

  TLegend *leg = new TLegend(0.6, 0.7, 0.9, 0.9);
  leg->SetFillStyle(0);
  leg->SetLineStyle(0);
  leg->SetTextSize(0.04);
  leg->SetLineColor(0);
  if (gTag_nn != "")
    leg->AddEntry(g_nn, gTag_nn, "L");
  if (gTag_bb != "")
    leg->AddEntry(g_bb, gTag_bb, "L");
  leg->Draw("same");

  c->SaveAs(path_output_graph + "_detailed_" + name_tag + ".json");
  c->Close();
  delete c;
}

void PlottingDetailed(TFile *file_nn, TFile *file_bb,
                      TString path_output_graph) {
  vector<TString> vec_tag_x = {"fTgl", "fFt0Occ"};
  if (gNoOccupancy) {
    // remove fFt0Occ if no occupancy is needed
    vec_tag_x.erase(remove(vec_tag_x.begin(), vec_tag_x.end(), "fFt0Occ"),
                    vec_tag_x.end());
  }
  vector<TString> vec_tag_ncls = {"highNcls", "lowNcls", "AllNcls"};

  for (const auto &tag_x : vec_tag_x)
    for (const auto &tag_ncls : vec_tag_ncls)
      PlottingDetailed(file_nn, file_bb, path_output_graph, tag_x, tag_ncls);
}

void PlottingSepPower(TFile *file_nn, TFile *file_bb,
                      TString path_output_graph) {
  vector<TString> vec_tag_x = {"fTgl", "fFt0Occ"};
  if (gNoOccupancy) {
    // remove fFt0Occ if no occupancy is needed
    vec_tag_x.erase(remove(vec_tag_x.begin(), vec_tag_x.end(), "fFt0Occ"),
                    vec_tag_x.end());
  }
  vector<TString> vec_tag_ncls = {"highNcls", "lowNcls", "AllNcls"};

  TCanvas *c_sepPower = new TCanvas("canvas_sepPower", "canvas_sepPower", 1200,
                                    gNoOccupancy ? 400 : 800);
  c_sepPower->Divide(3, gNoOccupancy ? 1 : 2);

  int index = 1;
  for (const auto &tag_x : vec_tag_x) {
    for (const auto &tag_ncls : vec_tag_ncls) {
      c_sepPower->cd(index++);
      gPad->SetTopMargin(0.1);
      TString name_tag = tag_x + "_sepPower_" + tag_ncls;
      TH1D *sepPower_nn = (TH1D *)file_nn->Get(name_tag);
      TH1D *sepPower_bb = (TH1D *)file_bb->Get(name_tag);
      TGraph *g_sepPower_nn = GetGraph(sepPower_nn, "g_" + name_tag + "_nn");
      TGraph *g_sepPower_bb = GetGraph(sepPower_bb, "g_" + name_tag + "_bb");
      MRootGraphic::StyleHistCommon(g_sepPower_nn);
      MRootGraphic::StyleHistCommon(g_sepPower_bb);
      g_sepPower_nn->SetLineColor(kRed);
      g_sepPower_bb->SetLineColor(kBlue);
      double max =
          TMath::Max(sepPower_nn->GetMaximum(), sepPower_bb->GetMaximum());
      double min =
          TMath::Min(sepPower_nn->GetMinimum(), sepPower_bb->GetMinimum());
      double max_user = max + 0.1 * (max - min);
      double min_user = min - 0.1 * (max - min);
      g_sepPower_nn->GetYaxis()->SetRangeUser(min_user, max_user);
      int index_graph = 0;
      if (gTag_nn != "") {
        if (index_graph == 0)
          g_sepPower_nn->Draw("AL");
        index_graph++;
      }
      if (gTag_bb != "") {
        if (index_graph == 0)
          g_sepPower_bb->Draw("AL");
        else
          g_sepPower_bb->Draw("L same");
      }

      if (index == 2) {
        TLegend *leg = new TLegend(0.5, 0.6, 0.7, 0.8);
        leg->SetFillStyle(0);
        leg->SetLineStyle(0);
        leg->SetLineColor(0);
        leg->SetTextSize(0.04);
        if (gTag_nn != "")
          leg->AddEntry(g_sepPower_nn, gTag_nn, "L");
        if (gTag_bb != "")
          leg->AddEntry(g_sepPower_bb, gTag_bb, "L");
        leg->Draw("same");
      }
    }
  }
  c_sepPower->SaveAs(path_output_graph + "_sepPower.json");
}

void ComparisonSeparationPower(
    TString path_input_nn = "~/test/sepPower_nn.root",
    TString path_input_bb = "~/test/sepPower_bb.root",
    TString path_output_graph = "~/test/sepPower_graph", TString tag_nn = "NN",
    TString tag_bb = "BB", bool no_occupancy = false) {
  TFile *file_nn = new TFile(path_input_nn, "READ");
  TFile *file_bb = new TFile(path_input_bb, "READ");

  gTag_bb = tag_bb;
  gTag_nn = tag_nn;

  MRootGraphic::StyleCommon();
  gStyle->SetPadTickX(1);
  gStyle->SetPadTickY(1);
  gStyle->SetPadGridX(1);
  gStyle->SetPadGridY(1);
  gNoOccupancy = no_occupancy;
  PlottingDetailed(file_nn, file_bb, path_output_graph);
  PlottingSepPower(file_nn, file_bb, path_output_graph);
}

int main(int argc, char **argv) {
  TString path_input_nn = "~/test/sepPower_nn.root";
  TString path_input_bb = "~/test/sepPower_bb.root";
  TString path_output_graph = "~/test/sepPower_graph";
  TString tag_nn = "NN";
  TString tag_bb = "BB";
  bool no_occupancy = false;

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
  if (argc > 6)
    no_occupancy = TString(argv[6]).Atoi();

  ComparisonSeparationPower(path_input_nn, path_input_bb, path_output_graph,
                            tag_nn, tag_bb, no_occupancy);
  return 0;
}