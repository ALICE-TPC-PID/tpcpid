#include "TBufferJSON.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TF1.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TKey.h"
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

using namespace std;

//====== pass4

// root.exe -l -b -q
// QAPIDTPC_V0.C'("/Users/tiantiancheng/Desktop/TPCPID_QA/LHC23h_pass4/AnalysisResults_merge_LHC23h_apass4.root",
// "LHC23h_pass4/TPCPIDQA_V0_LHC23h_pass4_BB.pdf")'

int GenerateUID() {
  static int uid = 0;
  return uid++;
}

TString gPath_output;

void SetupStyle();
TH2 *Get2DHistogramfromList(const char *dirname, TObject *histoname);
void AddFit(TH2 *h2d);

void PublishCanvas(TDirectoryFile *qaList);
void SetupPadStyle();
void LoadLibs();
Int_t CheckLoadLibrary(const char *library);

TCanvas *fCanvas = 0x0;

TString gTag_nnObb = "w/o NN";

void QAPIDTPC_V0(const char *inputFile =
                     "/home/szhu/test/AnalysisResults_merge_LHC23zzh.root",
                 TString outputFile = "/home/szhu/test/output/PID_V0.pdf",
                 TString tag_nnObb = "with NN") {

  gTag_nnObb = tag_nnObb;
  SetupStyle();

  TFile f(inputFile);
  if (!f.IsOpen()) {
    printf("Could not open file '%s'\n", f.GetName());
    return;
  }

  gPath_output = outputFile;

  fCanvas = new TCanvas;
  TPDF p(outputFile);

  std::string dirname;
  dirname = "track-pid-qa";
  TDirectoryFile *dir = (TDirectoryFile *)f.Get(dirname.c_str());

  PublishCanvas(dir);

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

// TH2* Get2DHistogramfromList(const char* dirname, TObject* histoname)
//{
//
//   TH2* histo = (TH2*)dirname->FindObject(histoname);
//   if (!histo) {printf(" histogram not found \n");  return 0x0; }
//
//   return histo;
// }

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

  TLine *l = 0x0;
  l = new TLine(h2d->GetXaxis()->GetXmin(), 0, h2d->GetXaxis()->GetXmax(), 0);
  l->SetLineStyle(2);
  h2d->GetListOfFunctions()->Add(l);
  l = new TLine(h2d->GetXaxis()->GetXmin(), 1, h2d->GetXaxis()->GetXmax(), 1);
  l->SetLineStyle(2);
  h2d->GetListOfFunctions()->Add(l);

} // AddFit

void PublishCanvas(TDirectoryFile *qaList) {
  TObjArray arrHistos;

  TPaveText pt(.1, .1, .9, .9, "NDC");
  pt.SetBorderSize(1);
  pt.SetFillColor(0);
  pt.SetTextSizePixels(16);

  pt.AddText("v0-pid-qa (" + gTag_nnObb + ")");
  // pt.AddText(Form("v0-pid-qa (" + gTag_nnObb + ")"));

  //  pt.AddText(Form("v0-pid-qa (w/o NN)"));
  arrHistos.Add(&pt);

  TList *listry = qaList->GetListOfKeys();

  TIter next(listry);
  TKey *key;
  TObject *obj;

  while ((key = (TKey *)next())) {
    obj = key->ReadObj();
    //         cout << " check the name of obj  " <<  obj->GetName() << endl;

    TString name_check = obj->GetName();
    //        Int_t Name_toKeep_Ele = strcmp(name_check, "h2TPCnSigma_Pin_El");
    //        Int_t Name_toKeep_Pi = strcmp(name_check, "h2TPCnSigma_Pin_Pi");
    //        Int_t Name_toKeep_Pr = strcmp(name_check, "h2TPCnSigma_Pin_Pr");
    //

    const char *objName = obj->GetName();
    //
    //       if( (Name_toKeep_Ele == 0 ) && ( Name_toKeep_Pi == 0 ) && (
    //       Name_toKeep_Pr == 0 ) ){
    //
    if (strcmp(objName, "h2TPCnSigma_Pin_El") == 0 ||
        strcmp(objName, "h2TPCnSigma_Pin_Pi") == 0 ||
        strcmp(objName, "h2TPCnSigma_Pin_Pr") == 0 ||
        strcmp(objName, "h2TPCnSigma_Pin_Ka") == 0) {

      TH2 *h = (TH2 *)qaList->FindObject(obj->GetName());

      cout << " check name :  " << name_check << endl;

      h->SetOption("colz");
      AddFit(h);
      arrHistos.Add(h);
      h->GetYaxis()->SetRangeUser(-5, 5);
      h->GetXaxis()->SetRangeUser(0.101, 30);
      h->GetYaxis()->SetTitle("n#sigma_{TPC}");
      h->GetXaxis()->SetTitle("p/|Z| (GeV/#it{c})");
      h->GetYaxis()->SetTitleSize(0.05);
      h->GetXaxis()->SetTitleSize(0.05);
      h->GetXaxis()->SetTitleOffset(0.8);
    }
  }

  Int_t nPads = arrHistos.GetEntriesFast();
  Int_t nCols = (Int_t)TMath::Ceil(TMath::Sqrt(nPads));
  Int_t nRows = (Int_t)TMath::Ceil((Double_t)nPads / (Double_t)nCols);
  fCanvas->Divide(nCols, nRows);

  for (Int_t i = 0; i < nPads; ++i) {
    fCanvas->cd(i + 1);
    SetupPadStyle();

    //      gPad->SetLogx(kFALSE);
    arrHistos.At(i)->Draw();
  }

  fCanvas->Update();

  TString path_json = gPath_output;
  // remove the extension
  path_json.Remove(path_json.Last('.'), path_json.Length());
  path_json.Append(Form("_%d.json", GenerateUID()));
  TBufferJSON::ExportToFile(path_json.Data(), fCanvas);
  fCanvas->SaveAs(Form("%s.root", gPath_output.Data()));
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
