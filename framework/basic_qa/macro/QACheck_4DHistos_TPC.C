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

using namespace std;
//===================================================================================
//
///    root.exe -l -b -q
///    QACheck_4DHistos_TPC.C'("/Users/tiantiancheng/Desktop/TPCPID_QA/LHC23h_pass3_QC/AnalysisResults_LHC23h_pass3_NN.root")'
///
//===================================================================================

// const TString output =
// "/Users/tiantiancheng/Desktop/TPCPID_QA/LHC23h_pass3_QC/QA_TPC/";

TString output =
    "/Users/tiantiancheng/Desktop/TPCPID_QA/LHC23h_pass3_QC/QA_TPCTOF/";

void SetupStyle();
TH2 *Get2DHistogramfromList(const char *dirname, TObject *histoname);
void AddFit(TH2 *h2d);

// void PublishCanvas(THnT<double> *qaList);

void SetupPadStyle();
void LoadLibs();
Int_t CheckLoadLibrary(const char *library);

TCanvas *fCanvas = 0x0;

// void QACheck_4DHistos (const char* inputFile, TString outputFile= "")

void QACheck_4DHistos_TPC(const char *inputFile = "/home/szhu/work/alice/tpc_pid/AutoQA/test/data/BB/AnalysisResults.root", TString outputFile = "/home/szhu/work/alice/tpc_pid/AutoQA/test/results2/BB/") {
  output = outputFile;
  LoadLibs();
  SetupStyle();

  TFile f(inputFile);
  if (!f.IsOpen()) {
    printf("Could not open file '%s'\n", f.GetName());
    return;
  }

  //---- read root file
  //    fCanvas=new TCanvas;
  //    TPDF p(outputFile);

  ///------- here TPC QA
#ifndef TPC_TOF
  std::string dirname;
  dirname = "tpc-pid-qa";
  TDirectoryFile *dir = (TDirectoryFile *)f.Get(dirname.c_str());

  printf("******* directory '%s' in file '%s' \n", dir->GetName(), f.GetName());

#else
  ///----------- here TPC + TOF QA
  std::string dirnameTPC;
  dirnameTPC = "tpc-pid-qa";
  TDirectoryFile *dirTPC = (TDirectoryFile *)f.Get(dirnameTPC.c_str());

  printf("******* directory '%s' in file '%s' \n", dirTPC->GetName(),
         f.GetName());

  std::string dirname;
  dirname = "wTOF";
  TDirectoryFile *dir = (TDirectoryFile *)dirTPC->Get(dirname.c_str());

  ////----------------
#endif

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

  ///----- THnT class
  THnSparseT<float> *h_FourD_Histos = nullptr;
  ///---------------------------------------------------------------
  /// THnT 4D: 0: PIn; 1: Eta; 2: NSigma; 3: NClsTPC
  ///-----------------------------------------------------------------

  ///----  Bin settings:
  Double_t Vary_TPCNcls[3] = {60, 130, 160};
  Double_t Vary_pT[15] = {0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0,
                          3.5, 4.0, 4.5, 5.0, 6.0, 8.0, 15.0};
  Double_t Vary_Eta[10] = {-0.9, -0.7, -0.5, -0.3, -0.1,
                           0.1,  0.3,  0.5,  0.7,  0.9};

  Double_t Mean_sigma_pr_pi[4][2][14][9];
  Double_t MeanError_sigma_pr_pi[4][2][14][9];

  //---- Cut settings
  Double_t fix_TPCnCls_Min, fix_TPCnCls_Max;
  Double_t fix_pT_Min, fix_pT_Max;
  Double_t fix_Eta_Min, fix_Eta_Max;
  TH1D *h_projection;
  Int_t Num_tracks[4][2][14][9];

  //--------------
  for (Int_t p = 0; p < 4; p++) {

    THnTname = THnT_filename[p];
    h_FourD_Histos =
        static_cast<THnSparseT<float> *>(subsubfile->Get(THnTname.c_str()));

    if (!h_FourD_Histos) {
      cerr << " *** fail to find h_FourD_Histos  " << endl;
    }

    ///------ do the projection based on the cuts
    for (Int_t i = 0; i < 2; i++) { // loop for TPCNCls range

      fix_TPCnCls_Min = Vary_TPCNcls[i];
      fix_TPCnCls_Max = Vary_TPCNcls[i + 1];

      for (Int_t j = 0; j < 14; j++) { // loop over pT range

        fix_pT_Min = Vary_pT[j];
        fix_pT_Max = Vary_pT[j + 1];

        for (Int_t k = 0; k < 9; k++) { // loop over Eta range

          fix_Eta_Min = Vary_Eta[k];
          fix_Eta_Max = Vary_Eta[k + 1];

          ///---- apply cuts
          h_FourD_Histos->GetAxis(3)->SetRangeUser(fix_TPCnCls_Min,
                                                   fix_TPCnCls_Max);
          h_FourD_Histos->GetAxis(0)->SetRangeUser(fix_pT_Min, fix_pT_Max);
          h_FourD_Histos->GetAxis(1)->SetRangeUser(fix_Eta_Min, fix_Eta_Max);

          ///------- create a dynamic hist names based on the loop index
          TString histName =
              Form("h_projection_%d_%d_%d_%s", i, j, k, THnTname.c_str());

          h_projection = (TH1D *)h_FourD_Histos->Projection(2, "e");
          h_projection->SetName(histName.Data());

          ///--------- use Gaussian function to do the fit
          TF1 *fitFunc = new TF1("fitFunc", "gaus");
          h_projection->Fit(fitFunc);
          Mean_sigma_pr_pi[p][i][j][k] = fitFunc->GetParameter(1);
          MeanError_sigma_pr_pi[p][i][j][k] = fitFunc->GetParError(1);

        } // k

      } // j

    } // i

    ///----- loop the subsubfile
  } /// loop the  folders!
    //
    //
  //    ////-------------------------------------------------------------------------
  //    ////--------------- Define and Fill histogram -------------------------
  //    ////-------------------------------------------------------------------------

  ////-------- hist: nSigma as a function of Eta

  TH1D *h_pi_pr_TPCNCls_PIn_nSigmaTPC_vs_Eta[4][2][14];
  for (Int_t p = 0; p < 4; p++) {
    THnTname = THnT_filename[p];

    for (Int_t i = 0; i < 2; i++) {

      fix_TPCnCls_Min = Vary_TPCNcls[i];
      fix_TPCnCls_Max = Vary_TPCNcls[i + 1];

      for (Int_t j = 0; j < 14; j++) {

        //---------------------------------
        //---- for pion and protons

        h_pi_pr_TPCNCls_PIn_nSigmaTPC_vs_Eta[p][i][j] =
            new TH1D(Form("h_eleTPCNCls_PIn_nSigmaTPC_vs_Eta_%s_%d_%d",
                          THnTname.c_str(), i, j),
                     "", 9, Vary_Eta);
        h_pi_pr_TPCNCls_PIn_nSigmaTPC_vs_Eta[p][i][j]->GetYaxis()->SetTitle(
            Form("mean of n#sigma_{TPC}^{%s}", THnTname.c_str()));
        h_pi_pr_TPCNCls_PIn_nSigmaTPC_vs_Eta[p][i][j]->GetXaxis()->SetTitle(
            "#eta");

        h_pi_pr_TPCNCls_PIn_nSigmaTPC_vs_Eta[p][i][j]->GetYaxis()->SetRangeUser(
            -5, 5);
        h_pi_pr_TPCNCls_PIn_nSigmaTPC_vs_Eta[p][i][j]->SetMarkerSize(0.5);
        h_pi_pr_TPCNCls_PIn_nSigmaTPC_vs_Eta[p][i][j]->GetYaxis()->SetTitleSize(
            0.05);
        h_pi_pr_TPCNCls_PIn_nSigmaTPC_vs_Eta[p][i][j]->GetXaxis()->SetTitleSize(
            0.05);
        h_pi_pr_TPCNCls_PIn_nSigmaTPC_vs_Eta[p][i][j]->GetYaxis()->SetLabelSize(
            0.04);
        h_pi_pr_TPCNCls_PIn_nSigmaTPC_vs_Eta[p][i][j]->GetXaxis()->SetLabelSize(
            0.04);

        h_pi_pr_TPCNCls_PIn_nSigmaTPC_vs_Eta[p][i][j]->SetTitle(
            Form("TPCnCls: %.f - %.f, %s (w/o NN)", fix_TPCnCls_Min,
                 fix_TPCnCls_Max, THnTname.c_str()));

        if (j < 7) {
          h_pi_pr_TPCNCls_PIn_nSigmaTPC_vs_Eta[p][i][j]->SetLineColor(kOrange +
                                                                      10 - j);
          h_pi_pr_TPCNCls_PIn_nSigmaTPC_vs_Eta[p][i][j]->SetMarkerColor(
              kOrange + 10 - j);
          h_pi_pr_TPCNCls_PIn_nSigmaTPC_vs_Eta[p][i][j]->SetMarkerStyle(24);
        } else {
          h_pi_pr_TPCNCls_PIn_nSigmaTPC_vs_Eta[p][i][j]->SetLineColor(kAzure -
                                                                      13 + j);
          h_pi_pr_TPCNCls_PIn_nSigmaTPC_vs_Eta[p][i][j]->SetMarkerColor(kAzure -
                                                                        13 + j);
          h_pi_pr_TPCNCls_PIn_nSigmaTPC_vs_Eta[p][i][j]->SetMarkerStyle(24);
        }
      } // j loop
    }
  }
  //
  //
  //    ///=========================================================================
  //    //-------- fill histos: nSigmaTPC_vs_Eta, with different pT bins
  //    ///=========================================================================
  //
  for (Int_t p = 0; p < 4; p++) {
    for (Int_t i = 0; i < 2; i++) {
      for (Int_t j = 0; j < 14; j++) {

        //--- loop for pion and proton
        for (Int_t k = 0; k < 9; k++) {

          h_pi_pr_TPCNCls_PIn_nSigmaTPC_vs_Eta[p][i][j]->SetBinContent(
              k + 1, Mean_sigma_pr_pi[p][i][j][k]);
          h_pi_pr_TPCNCls_PIn_nSigmaTPC_vs_Eta[p][i][j]->SetBinError(
              k + 1, MeanError_sigma_pr_pi[p][i][j][k]);
        }
      }
    }
  }

  //    ////---- Draw histos

  for (Int_t p = 1; p < 4; p++) {

    THnTname = THnT_filename[p];

    cout << " check particle name :  " << THnTname << endl;

    for (Int_t i = 0; i < 2; i++) {

      TCanvas *h1 = new TCanvas();
      h1->SetTickx(1);
      h1->SetTicky(1);
      h1->SetLeftMargin(0.12);
      h1->SetRightMargin(0.02);
      h1->SetTopMargin(0.1);
      h1->SetBottomMargin(0.1);

      TLegend *leg1 = new TLegend(0.18, 0.6, 0.95, 0.86);
      leg1->SetBorderSize(0);
      leg1->SetNColumns(3);

      fix_TPCnCls_Min = Vary_TPCNcls[i];
      fix_TPCnCls_Max = Vary_TPCNcls[i + 1];

      for (Int_t j = 0; j < 14; j++) {

        fix_pT_Min = Vary_pT[j];
        fix_pT_Max = Vary_pT[j + 1];

        //-- only draw pions and protons

        leg1->AddEntry(h_pi_pr_TPCNCls_PIn_nSigmaTPC_vs_Eta[p][i][j],
                       Form("p_{IN}: %.1f - %.1f ", fix_pT_Min, fix_pT_Max));
        h_pi_pr_TPCNCls_PIn_nSigmaTPC_vs_Eta[p][i][j]->Draw("e same");
      }

      leg1->Draw("same");

      //            h1 ->
      //            SaveAs(output+Form("h_nSigmaTPC_vs_Eta__TPCnCls_%.1f_%.1f__%s_TPC.png",
      //            fix_TPCnCls_Min, fix_TPCnCls_Max, THnTname.c_str()));

      h1->SaveAs(output +
                 Form("h_nSigmaTPC_vs_Eta__TPCnCls_%.1f_%.1f__%s_TPCTOF.png",
                      fix_TPCnCls_Min, fix_TPCnCls_Max, THnTname.c_str()));

      delete h1;
      delete leg1;
    }

  } // THnTname

  ////-----------------------------------------------------------------------------------------------
  ////--------------- Define and Fill histogram: nSigma_pT
  ///-----------------------------------------------
  ////-----------------------------------------------------------------------------------------------
  TH1D *h_pi_pr_TPCNCls_varyEta_nSigmaTPC_vs_PIn[4][2][9];

  //----------- for pion and proton
  for (Int_t p = 0; p < 4; p++) {

    THnTname = THnT_filename[p];

    for (Int_t i = 0; i < 2; i++) { //  loop for TPCNCls range

      fix_TPCnCls_Min = Vary_TPCNcls[i];
      fix_TPCnCls_Max = Vary_TPCNcls[i + 1];

      for (Int_t j = 0; j < 9; j++) { // loop over Eta range

        h_pi_pr_TPCNCls_varyEta_nSigmaTPC_vs_PIn[p][i][j] =
            new TH1D(Form("h_FixTPCNCls_varyEta_nSigmaTPC_vs_PIn_%s_%d_%d",
                          THnTname.c_str(), i, j),
                     "", 14, Vary_pT);
        h_pi_pr_TPCNCls_varyEta_nSigmaTPC_vs_PIn[p][i][j]->GetYaxis()->SetTitle(
            Form("mean of n#sigma_{TPC}^{%s}", THnTname.c_str()));
        h_pi_pr_TPCNCls_varyEta_nSigmaTPC_vs_PIn[p][i][j]->GetXaxis()->SetTitle(
            "P_{In}");

        h_pi_pr_TPCNCls_varyEta_nSigmaTPC_vs_PIn[p][i][j]
            ->GetYaxis()
            ->SetRangeUser(-7, 7);
        h_pi_pr_TPCNCls_varyEta_nSigmaTPC_vs_PIn[p][i][j]
            ->GetYaxis()
            ->SetTitleSize(0.05);
        h_pi_pr_TPCNCls_varyEta_nSigmaTPC_vs_PIn[p][i][j]
            ->GetXaxis()
            ->SetTitleSize(0.05);
        h_pi_pr_TPCNCls_varyEta_nSigmaTPC_vs_PIn[p][i][j]
            ->GetYaxis()
            ->SetLabelSize(0.04);
        h_pi_pr_TPCNCls_varyEta_nSigmaTPC_vs_PIn[p][i][j]
            ->GetXaxis()
            ->SetLabelSize(0.04);
        h_pi_pr_TPCNCls_varyEta_nSigmaTPC_vs_PIn[p][i][j]->SetMarkerSize(0.5);

        h_pi_pr_TPCNCls_varyEta_nSigmaTPC_vs_PIn[p][i][j]->SetTitle(
            Form("TPCnCls: %.f - %.f, %s (w/o NN)", fix_TPCnCls_Min,
                 fix_TPCnCls_Max, THnTname.c_str()));

        h_pi_pr_TPCNCls_varyEta_nSigmaTPC_vs_PIn[p][i][j]->SetLineColor(j + 1);
        h_pi_pr_TPCNCls_varyEta_nSigmaTPC_vs_PIn[p][i][j]->SetMarkerColor(j +
                                                                          1);
        h_pi_pr_TPCNCls_varyEta_nSigmaTPC_vs_PIn[p][i][j]->SetMarkerStyle(25);
      }
    }
  }
  //---- fill histograms: pions and protons

  for (Int_t p = 0; p < 4; p++) {
    for (Int_t i = 0; i < 2; i++) {
      for (Int_t j = 0; j < 9; j++) {

        for (Int_t k = 0; k < 14; k++) {
          h_pi_pr_TPCNCls_varyEta_nSigmaTPC_vs_PIn[p][i][j]->SetBinContent(
              k + 1, Mean_sigma_pr_pi[p][i][k][j]);
          h_pi_pr_TPCNCls_varyEta_nSigmaTPC_vs_PIn[p][i][j]->SetBinError(
              k + 1, MeanError_sigma_pr_pi[p][i][k][j]);
        }
      }
    }
  }

  //------ Draw histograms

  for (Int_t p = 1; p < 4; p++) {

    THnTname = THnT_filename[p];

    cout << " check particle name :  " << THnTname << endl;

    for (Int_t i = 0; i < 2; i++) {

      TCanvas *h2 = new TCanvas();
      h2->SetTickx(1);
      h2->SetTicky(1);
      h2->SetLeftMargin(0.12);
      h2->SetRightMargin(0.03);
      h2->SetTopMargin(0.1);
      h2->SetBottomMargin(0.12);

      TLegend *leg2 = new TLegend(0.18, 0.68, 0.95, 0.85);
      leg2->SetBorderSize(0);
      leg2->SetNColumns(3);

      fix_TPCnCls_Min = Vary_TPCNcls[i];
      fix_TPCnCls_Max = Vary_TPCNcls[i + 1];

      for (Int_t j = 0; j < 9; j++) {

        fix_Eta_Min = Vary_Eta[j];
        fix_Eta_Max = Vary_Eta[j + 1];

        leg2->AddEntry(h_pi_pr_TPCNCls_varyEta_nSigmaTPC_vs_PIn[p][i][j],
                       Form("#eta: %.1f  %.1f ", fix_Eta_Min, fix_Eta_Max));
        h_pi_pr_TPCNCls_varyEta_nSigmaTPC_vs_PIn[p][i][j]->Draw("e same");
      }

      leg2->Draw("same");

      //      h2 ->
      //      SaveAs(output+Form("h_nSigmaTPC_vs_pT__TPCnCls_%.1f_%.1f__%s_TPC.png",
      //      fix_TPCnCls_Min, fix_TPCnCls_Max, THnTname.c_str()));

      h2->SaveAs(output +
                 Form("h_nSigmaTPC_vs_pT__TPCnCls_%.1f_%.1f__%s_TPCTOF.png",
                      fix_TPCnCls_Min, fix_TPCnCls_Max, THnTname.c_str()));

      delete h2;
      delete leg2;

    } // TPCNCls

  } // particles

  ///*****************************
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
  st->SetTitleBorderSize(0);
  st->SetTitleStyle(0);

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
