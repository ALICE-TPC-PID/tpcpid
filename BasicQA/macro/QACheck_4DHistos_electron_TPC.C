#include "TCanvas.h"
#include "TColor.h"
#include "TF1.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "THashList.h"
#include "THn.h"
#include "THnSparse.h"
#include "TLegend.h"
#include "TLine.h"
#include "TList.h"
#include "TMath.h"
#include "TObjArray.h"
#include "TPDF.h"
#include "TPaveText.h"
#include "TROOT.h"
#include "TString.h"
#include "TStyle.h"
#include "TSystem.h"
#include <array>
#include <fstream>
#include <sstream>

// #include "./MACRO_CONTROL.h"

//===================================================================================
//
///     root.exe -l -b -q
///     QACheck_4DHistos_electron_TPC.C'("/Users/tiantiancheng/Desktop/TPCPID_QA/LHC23zs_apass2_QC1_sampling/AnalysisResults_LHC23zs_apass2_QC1_Sampling.root")'
///
//===================================================================================

// const TString output =
// "/Users/tiantiancheng/Desktop/TPCPID_QA/LHC23zs_apass2_QC1_sampling/QA_TPC/Fit_Projection_forElectron/";

using namespace std;

TString output =
    "/Users/tiantiancheng/Desktop/TPCPID_QA/LHC23zs_apass2_QC1_sampling/"
    "QA_TPCTOF/Fit_Projection_forElectron/";

void SetupStyle();
TH2 *Get2DHistogramfromList(const char *dirname, TObject *histoname);
void AddFit(TH2 *h2d);

void PublishCanvas(THnT<double> *qaList);

void SetupPadStyle();
void LoadLibs();
Int_t CheckLoadLibrary(const char *library);

TCanvas *fCanvas = 0x0;

// void QACheck_4DHistos (const char* inputFile, TString outputFile= "")

void QACheck_4DHistos_electron_TPC(const char *inputFile, TString outputFile) {
  output = outputFile;
  LoadLibs();
  SetupStyle();

  TFile f(inputFile);
  if (!f.IsOpen()) {
    printf("Could not open file '%s'\n", f.GetName());
    return;
  }

  ///------ here TPC QA
#ifndef TPC_TOF
  std::string dirname;
  dirname = "tpc-pid-qa";
  TDirectoryFile *dir = (TDirectoryFile *)f.Get(dirname.c_str());
  printf("******* directory '%s' in file '%s' \n", dir->GetName(), f.GetName());
#else
  ///-------- here TPC + TOF QA
  std::string dirnameTPC;
  dirnameTPC = "tpc-pid-qa";
  TDirectoryFile *dirTPC = (TDirectoryFile *)f.Get(dirnameTPC.c_str());

  printf("******* directory '%s' in file '%s' \n", dirTPC->GetName(),
         f.GetName());

  std::string dirname;
  dirname = "wTOF";
  TDirectoryFile *dir = (TDirectoryFile *)dirTPC->Get(dirname.c_str());
#endif
  ///---------------------
  std::string subdirname;
  subdirname = "nsigma";
  TDirectoryFile *subfile = (TDirectoryFile *)dir->Get(subdirname.c_str());

  std::string subsubdirname;
  subsubdirname = "sparsePinEtaNcl";
  TDirectoryFile *subsubfile =
      (TDirectoryFile *)subfile->Get(subsubdirname.c_str());

  if (!subsubfile) {
    cerr << " Can not find subsub file ******  " << endl;
    exit(1);
  }

  subsubfile->ls();

  ///--- here to get the THnSparse
  const char *THnT_filename[] = {"El", "Pi", "Ka", "Pr"};
  std::string THnTname;
  THnTname = "El";
  //    ///----- THnT class
  //    THnSparseT<float> *h_FourD_Histos = nullptr;
  //
  ///---------------------------------------------------------------
  /// THnT 4D:  0: PIn; 1: Eta; 2: NSigma; 3: NClsTPC
  ///-----------------------------------------------------------------

  ///----  Bin settings:
  Double_t Vary_TPCNcls[3] = {60, 130, 160};
  Double_t Vary_pT[9] = {0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 5.0};
  Double_t Vary_Eta[10] = {-0.9, -0.7, -0.5, -0.3, -0.1,
                           0.1,  0.3,  0.5,  0.7,  0.9};

  Double_t Mean_sigma_ele[4][14][9];
  Double_t MeanError_sigma_ele[4][14][9];

  //---- Cut settings
  Double_t fix_TPCnCls_Min, fix_TPCnCls_Max;
  Double_t fix_pT_Min, fix_pT_Max;
  Double_t fix_Eta_Min, fix_Eta_Max;
  TH1D *h_projection;

  THnSparseT<float> *h_FourD_Histos = nullptr;
  h_FourD_Histos =
      static_cast<THnSparseT<float> *>(subsubfile->Get(THnTname.c_str()));

  cout << "***** check the name:  " << h_FourD_Histos->GetName() << endl;
  if (!h_FourD_Histos) {
    cerr << " ***** can not find h_FourD_Histos " << endl;
  }

  //---- start to do the projection

  Int_t Num_tracks[2][9][8];

  for (Int_t i = 0; i < 2; i++) { // loop for TPCNCls range

    fix_TPCnCls_Min = Vary_TPCNcls[i];
    fix_TPCnCls_Max = Vary_TPCNcls[i + 1];

    for (Int_t j = 0; j < 9; j++) { // loop over Eta range

      fix_Eta_Min = Vary_Eta[j];
      fix_Eta_Max = Vary_Eta[j + 1];

      for (Int_t k = 0; k < 8; k++) { // loop over pT range

        fix_pT_Min = Vary_pT[k];
        fix_pT_Max = Vary_pT[k + 1];

        ///---- apply cuts
        h_FourD_Histos->GetAxis(3)->SetRangeUser(fix_TPCnCls_Min,
                                                 fix_TPCnCls_Max);
        h_FourD_Histos->GetAxis(0)->SetRangeUser(fix_pT_Min, fix_pT_Max);
        h_FourD_Histos->GetAxis(1)->SetRangeUser(fix_Eta_Min, fix_Eta_Max);

        ///------- create a dynamic hist names based on the loop index
        TString histName = Form("h_projection_%d_%d_%d", i, j, k);

        h_projection = (TH1D *)h_FourD_Histos->Projection(2, "e");
        h_projection->SetName(histName.Data());

        ///---- single gaussian to fit
        //                TF1 *fitFunc = new TF1("fitFunc", "gaus");
        //                h_projection -> Fit(fitFunc);
        //
        //                Mean_sigma_ele[i][j][k] = fitFunc -> GetParameter(1);
        //                MeanError_sigma_ele[i][j][k] = fitFunc ->
        //                GetParError(1);

        ///--- use double Gaussian to fit electron, and check the ampltidue
        ////
        TF1 *fitFunc = new TF1("fitFunc", "gaus(0)+gaus(3)");
        fitFunc->SetLineColor(kGreen + 2);
        fitFunc->SetParameter(0, 1.53589e+07);
        fitFunc->SetParameter(1, -5.13187e+00);
        fitFunc->SetParameter(2, 8.12928e-01);
        fitFunc->SetParameter(3, 7.63898e+02);
        fitFunc->SetParameter(4, -1.07983e-02);
        fitFunc->SetParameter(5, 1.01862e+00);

        h_projection->Fit(fitFunc);
        //
        //                ////------ draw the hprojection after double gaussian
        //                fitting procedure
        TCanvas *c_p = new TCanvas();
        c_p->SetTickx(1);
        c_p->SetTicky(1);
        h_projection->Draw();
        c_p->SaveAs(output +
                    Form("h_projection_withFit_TPCnCls_%.f_To_%.f__EtaRange_%."
                         "1f_To_%.1f__pTRange_%.1f_To_%.1f__.png",
                         fix_TPCnCls_Min, fix_TPCnCls_Max, fix_Eta_Min,
                         fix_Eta_Max, fix_pT_Min, fix_pT_Max));
        //
        ///--- here have to compare the amplitude, choose the main gaussian
        if (fitFunc->GetParameter(3) < fitFunc->GetParameter(0)) {
          Mean_sigma_ele[i][j][k] = fitFunc->GetParameter(1);
          MeanError_sigma_ele[i][j][k] = fitFunc->GetParError(1);

        } else {
          Mean_sigma_ele[i][j][k] = fitFunc->GetParameter(4);
          MeanError_sigma_ele[i][j][k] = fitFunc->GetParError(4);
        }
        ////--------------------

      } // k: pT

    } // j: Eta

  } // i:TPCNCls

  //-----------------------------------------------------------------------------------------------
  //--------------- Define and Fill histogram: nSigma_ Eta
  //------------------------------------------
  //-----------------------------------------------------------------------------------------------
  ////  nSigma vs Eta
  TH1D *h_ele_TPCNCls_PIn_nSigmaTPC_vs_Eta[2][8];

  for (Int_t i = 0; i < 2; i++) { //  loop for TPCNCls range

    fix_TPCnCls_Min = Vary_TPCNcls[i];
    fix_TPCnCls_Max = Vary_TPCNcls[i + 1];

    for (Int_t j = 0; j < 8; j++) { // loop over pT range

      h_ele_TPCNCls_PIn_nSigmaTPC_vs_Eta[i][j] =
          new TH1D(Form("h_FixTPCNCls_varyEta_nSigmaTPC_vs_PIn_%d_%d", i, j),
                   "", 9, Vary_Eta);
      h_ele_TPCNCls_PIn_nSigmaTPC_vs_Eta[i][j]->GetYaxis()->SetTitle(
          Form("mean of n#sigma_{TPC}^{%s}", THnTname.c_str()));
      h_ele_TPCNCls_PIn_nSigmaTPC_vs_Eta[i][j]->GetXaxis()->SetTitle("#eta");

      h_ele_TPCNCls_PIn_nSigmaTPC_vs_Eta[i][j]->GetYaxis()->SetRangeUser(-3, 3);
      h_ele_TPCNCls_PIn_nSigmaTPC_vs_Eta[i][j]->SetMarkerSize(0.5);

      h_ele_TPCNCls_PIn_nSigmaTPC_vs_Eta[i][j]->SetTitle(
          Form("TPCnCls: %.f - %.f, %s", fix_TPCnCls_Min, fix_TPCnCls_Max,
               THnTname.c_str()));

      h_ele_TPCNCls_PIn_nSigmaTPC_vs_Eta[i][j]->SetLineColor(j + 1);
      h_ele_TPCNCls_PIn_nSigmaTPC_vs_Eta[i][j]->SetMarkerColor(j + 1);
      h_ele_TPCNCls_PIn_nSigmaTPC_vs_Eta[i][j]->SetMarkerStyle(25);
    }

  } // i: TPCNCls

  ///------- fill histogram

  for (Int_t i = 0; i < 2; i++) {

    for (Int_t j = 0; j < 8; j++) {

      for (Int_t k = 0; k < 9; k++) {

        h_ele_TPCNCls_PIn_nSigmaTPC_vs_Eta[i][j]->SetBinContent(
            k + 1, Mean_sigma_ele[i][k][j]);
        h_ele_TPCNCls_PIn_nSigmaTPC_vs_Eta[i][j]->SetBinError(
            k + 1, MeanError_sigma_ele[i][k][j]);
      }
    }
  }

  ///---- Draw histogram

  for (Int_t i = 0; i < 2; i++) {

    TCanvas *h2 = new TCanvas();
    h2->SetTickx(1);
    h2->SetTicky(1);

    TLegend *leg2 = new TLegend(0.15, 0.7, 0.9, 0.9);
    leg2->SetBorderSize(0);
    leg2->SetNColumns(3);

    fix_TPCnCls_Min = Vary_TPCNcls[i];
    fix_TPCnCls_Max = Vary_TPCNcls[i + 1];

    for (Int_t j = 0; j < 8; j++) {

      fix_pT_Min = Vary_pT[j];
      fix_pT_Max = Vary_pT[j + 1];

      leg2->AddEntry(h_ele_TPCNCls_PIn_nSigmaTPC_vs_Eta[i][j],
                     Form("PIn: %.2f - %.2f ", fix_pT_Min, fix_pT_Max));
      h_ele_TPCNCls_PIn_nSigmaTPC_vs_Eta[i][j]->Draw("e same");
      leg2->Draw("same");
    }

    ///   h2 ->
    ///   SaveAs(output+Form("h_nSigmaTPC_vs_Eta__TPCnCls_%.1f_%.1f__%s_TPC.png",
    ///   fix_TPCnCls_Min, fix_TPCnCls_Max, THnTname.c_str()));

    h2->SaveAs(output +
               Form("h_nSigmaTPC_vs_Eta__TPCnCls_%.1f_%.1f__%s_TPCTOF.png",
                    fix_TPCnCls_Min, fix_TPCnCls_Max, THnTname.c_str()));

    delete h2;
    delete leg2;
  }

  ////-----------------------------------------------------------------------------------------------
  ////--------------- Define and Fill histogram: nSigma_ pT
  ///------------------------------------------
  ////-----------------------------------------------------------------------------------------------

  TH1D *h_ele_TPCNCls_Eta_nSigmaTPC_vs_PIn[2][9];

  for (Int_t i = 0; i < 2; i++) { //  loop for TPCNCls range

    fix_TPCnCls_Min = Vary_TPCNcls[i];
    fix_TPCnCls_Max = Vary_TPCNcls[i + 1];

    for (Int_t j = 0; j < 9; j++) { // loop over Eta range

      h_ele_TPCNCls_Eta_nSigmaTPC_vs_PIn[i][j] =
          new TH1D(Form("h_ele_TPCNCls_Eta_nSigmaTPC_vs_PIn%d_%d", i, j), "", 8,
                   Vary_pT);
      h_ele_TPCNCls_Eta_nSigmaTPC_vs_PIn[i][j]->GetYaxis()->SetTitle(
          Form("mean of n#sigma_{TPC}^{%s}", THnTname.c_str()));
      h_ele_TPCNCls_Eta_nSigmaTPC_vs_PIn[i][j]->GetXaxis()->SetTitle("PIn");

      h_ele_TPCNCls_Eta_nSigmaTPC_vs_PIn[i][j]->GetYaxis()->SetRangeUser(-3, 3);
      h_ele_TPCNCls_Eta_nSigmaTPC_vs_PIn[i][j]->SetMarkerSize(0.5);

      h_ele_TPCNCls_Eta_nSigmaTPC_vs_PIn[i][j]->SetTitle(
          Form("TPCnCls: %.f - %.f, %s", fix_TPCnCls_Min, fix_TPCnCls_Max,
               THnTname.c_str()));

      h_ele_TPCNCls_Eta_nSigmaTPC_vs_PIn[i][j]->SetLineColor(j + 1);
      h_ele_TPCNCls_Eta_nSigmaTPC_vs_PIn[i][j]->SetMarkerColor(j + 1);
      h_ele_TPCNCls_Eta_nSigmaTPC_vs_PIn[i][j]->SetMarkerStyle(25);
    }

  } // i: TPCNCls

  //---- fill histogram
  for (Int_t i = 0; i < 2; i++) {

    for (Int_t j = 0; j < 9; j++) {

      for (Int_t k = 0; k < 8; k++) {

        h_ele_TPCNCls_Eta_nSigmaTPC_vs_PIn[i][j]->SetBinContent(
            k + 1, Mean_sigma_ele[i][j][k]);
        h_ele_TPCNCls_Eta_nSigmaTPC_vs_PIn[i][j]->SetBinError(
            k + 1, MeanError_sigma_ele[i][j][k]);
      }
    }
  }

  //--- draw histogram

  ///---- Draw histogram
  for (Int_t i = 0; i < 2; i++) {

    TCanvas *h1 = new TCanvas();
    h1->SetTickx(1);
    h1->SetTicky(1);

    TLegend *leg1 = new TLegend(0.15, 0.7, 0.85, 0.9);
    leg1->SetBorderSize(0);
    leg1->SetNColumns(3);

    fix_TPCnCls_Min = Vary_TPCNcls[i];
    fix_TPCnCls_Max = Vary_TPCNcls[i + 1];

    for (Int_t j = 0; j < 9; j++) {

      fix_Eta_Min = Vary_Eta[j];
      fix_Eta_Max = Vary_Eta[j + 1];

      leg1->AddEntry(h_ele_TPCNCls_Eta_nSigmaTPC_vs_PIn[i][j],
                     Form("#eta: %.1f  %.1f ", fix_Eta_Min, fix_Eta_Max));
      h_ele_TPCNCls_Eta_nSigmaTPC_vs_PIn[i][j]->Draw("e same");
      leg1->Draw("same");
    }

    ///   h1 ->
    ///   SaveAs(output+Form("h_nSigmaTPC_vs_pT__TPCnCls_%.1f_%.1f__%s_TPC.png",
    ///   fix_TPCnCls_Min, fix_TPCnCls_Max, THnTname.c_str()));

    h1->SaveAs(output +
               Form("h_nSigmaTPC_vs_pT__TPCnCls_%.1f_%.1f__%s_TPCTOF.png",
                    fix_TPCnCls_Min, fix_TPCnCls_Max, THnTname.c_str()));

    delete h1;
    delete leg1;
  }

} // main

void SetupStyle() {
  const Int_t NCont = 255;

  TStyle *st = new TStyle("mystyle", "mystyle");
  gROOT->GetStyle("Plain")->Copy((*st));
  st->SetTitleX(0.1);
  st->SetTitleW(0.8);
  st->SetTitleH(0.08);
  st->SetStatX(.9);
  st->SetStatY(.9);
  st->SetNumberContours(NCont);
  st->SetPalette(1, 0);
  st->SetOptStat("erm");
  st->SetOptFit(0);
  st->SetGridColor(kGray + 1);
  st->SetPadGridX(kTRUE);
  st->SetPadGridY(kTRUE);
  st->SetPadTickX(kTRUE);
  st->SetPadTickY(kTRUE);
  st->cd();
  st->SetOptStat(0);

  const Int_t NRGBs = 5;
  Double_t stops[NRGBs] = {0.00, 0.34, 0.61, 0.84, 1.00};
  Double_t red[NRGBs] = {0.00, 0.00, 0.87, 1.00, 0.51};
  Double_t green[NRGBs] = {0.00, 0.81, 1.00, 0.20, 0.00};
  Double_t blue[NRGBs] = {0.51, 1.00, 0.12, 0.00, 0.00};

  TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);

} // SetupStyle

void SetupPadStyle() {
  //  gPad->SetLogx();
  gPad->SetLogz();
  gPad->SetGridx();
  gPad->SetGridy();
  gPad->SetTickx();
  gPad->SetTicky();
  // gStyle->SetErrorY(0);
} // SetupPadStyle

void LoadLibs() {
  CheckLoadLibrary("libCore");
  CheckLoadLibrary("libPhysics");
  CheckLoadLibrary("libMinuit");
  CheckLoadLibrary("libGui");
  CheckLoadLibrary("libXMLParser");
  CheckLoadLibrary("libGeom");
  //  CheckLoadLibrary("libVMC");
  CheckLoadLibrary("libNet");
  CheckLoadLibrary("libTree");
  CheckLoadLibrary("libProof");
  //  CheckLoadLibrary("libSTEERBase");
  //  CheckLoadLibrary("libESD");
  //  CheckLoadLibrary("libCDB");
  //  CheckLoadLibrary("libRAWDatabase");
  //  CheckLoadLibrary("libRAWDatarec");
  //  CheckLoadLibrary("libANALYSIS");
  //  CheckLoadLibrary("libSTEER");
  //  CheckLoadLibrary("libSTAT");
  //  CheckLoadLibrary("libAOD");
  //  CheckLoadLibrary("libOADB");
  //  CheckLoadLibrary("libANALYSISalice");
  //  CheckLoadLibrary("libCORRFW");
  //  CheckLoadLibrary("libTPCbase");

} // LoadLibs

Int_t CheckLoadLibrary(const char *library) {
  // checks if a library is already loaded, if not loads the library

  if (strlen(gSystem->GetLibraries(library, "", kFALSE)) > 0)
    return 1;
  return gSystem->Load(library);

} // CheckLoadLibrary
