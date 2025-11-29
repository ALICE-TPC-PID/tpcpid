#include "TCanvas.h"
#include "TColor.h"
#include "TF1.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "THashList.h"
#include "THn.h"
#include "THnSparse.h"
#include "TKey.h"
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
#include "TTree.h"
#include <array>
#include <fstream>
#include <sstream>
using namespace std;

//====== pass4

/// root.exe -l -b -q
/// QAPIDqaReport_optimized.C'("LHC23h_pass3_QC/AnalysisResults_LHC23h_pass3_NN.root",
/// "LHC23h_pass3_QC/TPCPIDQA_TPCTOF_LHC23h_pass3_QC1_w/o NN.pdf")'
///
///
////// root.exe -l -b -q
///QAPIDqaReport_optimized.C'("LHC23h_pass4/AnalysisResults_merge_LHC23h_apass4.root",
///"LHC23h_pass4/TPCPIDQA_TPCTOF_LHC23h_pass4_BB.pdf")'

void SetupStyle();
TH2 *Get2DHistogramfromList(TDirectoryFile *pidqalist, const char *subdirname,
                            TObject *histoname);
void AddFit(TH2 *h2d);

void PublishCanvas(TDirectoryFile *qaList, const char *subdirname);
void SetupPadStyle();
void LoadLibs();
Int_t CheckLoadLibrary(const char *library);

TCanvas *fCanvas = 0x0;

void QAPIDqaReport_optimized(const char *inputFile, TString outputFile = "") {

  LoadLibs();
  SetupStyle();

  TFile f(inputFile);
  if (!f.IsOpen()) {
    printf("Could not open file '%s'\n", f.GetName());
    return;
  }

  fCanvas = new TCanvas;
  TPDF p(outputFile);

  std::string dirname;
  TDirectoryFile *qaList = nullptr;

  //    //--- tpc-pid-qa
  dirname = "tpc-pid-qa";
  qaList = (TDirectoryFile *)f.Get(dirname.c_str());
  if (!qaList) {
    printf("Could not find directory '%s' in file '%s' \n", dirname.c_str(),
           f.GetName());
    return;
  }
  PublishCanvas(qaList, "nsigma");

  //  //---- Add wTOF
  std::string dirname1;
  dirname1 = "wTOF";
  TDirectoryFile *qaList1 = nullptr;
  qaList1 = (TDirectoryFile *)qaList->Get(dirname1.c_str());

  if (!qaList1) {
    printf("Could not find list '%s' in file '%s' \n ", dirname1.c_str(),
           qaList->GetName());
    return;
  }
  PublishCanvas(qaList1, "nsigma");
  delete qaList1;

  ////----- tof-pid-qa

  //    dirname = "tof-pid-qa";
  //    qaList = (TDirectoryFile*)f.Get(dirname.c_str());
  //    if (!qaList) {
  //      printf("Could not find directory '%s' in file '%s' \n",
  //      dirname.c_str(), f.GetName()); return false;
  //    }
  //
  //
  //    cout << " ************** check nSigma directory ******************  " <<
  //    endl; qaList -> ls();
  //
  //
  //    PublishCanvas(qaList, "nsigma");
  //
  //

  delete qaList;

  p.Close();
  delete fCanvas;

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

TH2 *Get2DHistogramfromList(TDirectoryFile *pidqalist, const char *subdirname,
                            TObject *histoname) {

  TDirectoryFile *histolist = (TDirectoryFile *)pidqalist->Get(subdirname);
  if (!histolist) {
    printf(" directory not found \n");
    return 0x0;
  }

  TH2 *histo = (TH2 *)histolist->FindObject(histoname);
  if (!histo) {
    printf(" histogram not found \n");
    return 0x0;
  }

  return histo;
}

void AddFit(TH2 *h2d) {
  //
  // Fit in slices and draw mean and sigma
  //

  TF1 *f1 = new TF1("f1", "gaus");
  f1->SetRange(-2, 2);
  TObjArray aSlices;

  h2d->FitSlicesY(f1, 0, -1, 0, "QNR", &aSlices);
  aSlices.SetOwner(1);

  //--- Three objects: Mean, Sigma, Chi2
  TH1 *hMean = (TH1 *)aSlices.At(1);
  TH1 *hSigma = (TH1 *)aSlices.At(2);
  // TH1* hChi2=(TH1*)aSlices.At(3);

  // hChi2->Scale(1./10.);
  aSlices.AddAt(0x0, 1);
  aSlices.AddAt(0x0, 2);
  // aSlices.AddAt(0x0,3);

  hMean->SetMarkerStyle(20);
  hMean->SetMarkerSize(0.3);
  hMean->SetOption("same hist p");
  h2d->GetListOfFunctions()->Add(hMean);

  hSigma->SetMarkerStyle(20);
  hSigma->SetMarkerSize(0.3);
  hSigma->SetOption("same hist p");
  hSigma->SetMarkerColor(kMagenta);
  h2d->GetListOfFunctions()->Add(hSigma);

  // hChi2->SetOption("same");
  // hChi2->SetMarkerColor(kMagenta + 2);
  // hChi2->SetLineColor(kMagenta + 2);
  // h2d->GetListOfFunctions()->Add(hChi2);

  TLine *l = 0x0;
  l = new TLine(h2d->GetXaxis()->GetXmin(), 0, h2d->GetXaxis()->GetXmax(), 0);
  l->SetLineStyle(2);
  h2d->GetListOfFunctions()->Add(l);
  l = new TLine(h2d->GetXaxis()->GetXmin(), 1, h2d->GetXaxis()->GetXmax(), 1);
  l->SetLineStyle(2);
  h2d->GetListOfFunctions()->Add(l);

} // AddFit

void PublishCanvas(TDirectoryFile *qaList, const char *subdirname) {
  TObjArray arrHistos;

  TPaveText pt(.1, .1, .9, .9, "NDC");
  pt.SetBorderSize(1);
  pt.SetFillColor(0);
  pt.SetTextSizePixels(16);

  pt.AddText(Form("%s %s", qaList->GetName(), subdirname));
  arrHistos.Add(&pt);

  TDirectoryFile *QA = (TDirectoryFile *)qaList->Get(subdirname);
  TList *listry = QA->GetListOfKeys();

  //    cout << " ******* check list ******* " << endl;
  //    QA -> ls();

  TIter next(listry);
  TKey *key;
  TObject *obj;

  while ((key = (TKey *)next())) {
    obj = key->ReadObj();
    //
    //        cout << " ************ qa list name   ***** "   <<
    //        qaList->GetName() << endl; cout << " ************ check the name
    //        of obj  ***** " <<  obj->GetName() << endl;
    // if ( strcmp(obj->GetName(),"pt") ==1 ) continue;

    //
    TString name_check = obj->GetName();
    Int_t Name_toRemove = strcmp(name_check, "pt");
    Int_t Name_toRemovemu = strcmp(name_check, "Mu");
    Int_t Name_toRemoveDe = strcmp(name_check, "De");
    Int_t Name_toRemoveTr = strcmp(name_check, "Tr");
    Int_t Name_toRemoveHe = strcmp(name_check, "He");
    Int_t Name_toRemoveAl = strcmp(name_check, "Al");
    Int_t Name_toRemove_Sparse = strcmp(name_check, "sparsePinEtaNcl");
    //
    ////        cout << "check the function of strcmp  "  <<Name_toRemove <<
    ///endl;
    ////
    if ((Name_toRemove != 0) && (Name_toRemovemu != 0) &&
        (Name_toRemoveDe != 0) && (Name_toRemoveTr != 0) &&
        (Name_toRemoveHe != 0) && (Name_toRemoveAl != 0) &&
        (Name_toRemove_Sparse != 0)) {
      //
      //             if( (Name_toRemove != 0)   &&  (Name_toRemovemu != 0 ) &&
      //             (Name_toRemoveDe !=0 ) && (Name_toRemoveTr != 0 ) && (
      //             Name_toRemoveHe != 0 ) && ( Name_toRemoveAl != 0 )   ){
      //    if( (Name_toRemove != 0) ){

      //        cout << " ***********  check name  " << name_check << endl;
      // if (strcmp(name_check, "nsigmaTPCV0VsEta") ==1 ) continue;
      //       // if(  name_check ==  nameInDir) continue;
      //
      //        if (strcmp(name_check, "pt") == 0) // only have pt
      //
      ///        cout << " qa list name"   <<  qaList->GetName() << endl;
      // cout << " check the name of obj  " <<  obj->GetName() << endl;
      //
      TH2 *h = Get2DHistogramfromList(qaList, subdirname, obj);
      h->SetOption("colz");
      AddFit(h);
      arrHistos.Add(h);
      h->GetYaxis()->SetRangeUser(-5, 5);
      h->GetXaxis()->SetRangeUser(0.101, 30);

      //
    } // loop to remove name
  }

  Int_t nPads = arrHistos.GetEntriesFast();
  Int_t nCols = (Int_t)TMath::Ceil(TMath::Sqrt(nPads));
  Int_t nRows = (Int_t)TMath::Ceil((Double_t)nPads / (Double_t)nCols);
  fCanvas->Divide(nCols, nRows);

  for (Int_t i = 0; i < nPads; ++i) {
    fCanvas->cd(i + 1);
    SetupPadStyle();
    if (strcmp(subdirname, "nsigmaTPCV0VsEta") == 0)
      gPad->SetLogx(kFALSE);
    arrHistos.At(i)->Draw();
  }

  fCanvas->Update();
  fCanvas->Clear();

} // PublishCanvas

void SetupPadStyle() {
  gPad->SetLogx();
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
