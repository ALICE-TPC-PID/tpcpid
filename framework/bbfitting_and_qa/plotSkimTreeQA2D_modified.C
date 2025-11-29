/*------------------------------------------------------------------
Brief Introduction:
This macro creates the QA plots from the TPC-V0 Skimmed tree.
The tree should be a merged skimmed tree. Althout it would work
on any test skimmed tree (or any other tree if the Branch names
are exactly same).
----------------
How to Run:
root -l  plotMacroSkimQA2.C
in the same directory where you have the merged skimmed tree file.
Note: Make a subdirectory named as "outputFigures" in the same
work directory where you are running the macro. This would save the
pdf files in the outputFigures directory.                            */
//_____________________rihanphys@gmail.com____________________________

// This macro was modified to automatically use the provided pathtoskimtree when entering the name (f.e. LHC23zzk)

#include "../utils/headerfunction.h"
#include "read_config.C"

///Global Parameter for Canvas:
Int_t  xpos=20, ypos=20;  //Default Canvas starting Position. Shifted with xpos += <val>; (see later in macro)
Int_t  cansizeX=500;      //Default Canvas Size X pixels
Int_t  cansizeY=580;      //Default Canvas Size Y pixels
Float_t xLabelPos = 0.5;  //Default Position of Label position.
TLine   *line;
TLatex  *tex;
TLegend *legend=0x0;


//Declare Functions:
TGraphErrors* getGraphMeanYFrom2DHist(TH2 *hNSigTPCpos=0x0,Float_t yLow=-1E2,Float_t yHigh=1E3);
void plotdEdxvsEta(TH2 *hNSigTPCpos=0x0, TH2 *hNSigTPCneg=0x0, TString name="", Float_t fMass=1.0, TString passName="",Float_t ySLow=0.1, Float_t ySHigh=2E2, Bool_t Print=kFALSE);
void plotNsigmaVsBG(TH2 *hNSigTPCPos=0x0, TH2 *hNSigTPCNeg=0x0, TH2 *hNSigTOFPos=0x0, TH2 *hNSigTOFNeg=0x0, TString name="", Float_t fMass=1.0, TString passName="",Float_t ySLow=0.1, Float_t ySHigh=1E4,Bool_t Print=kFALSE);
void plotNsigmaVsPin(TH2 *hNSigTPCPos=0x0, TH2 *hNSigTPCNeg=0x0, TH2 *hNSigTOFPos=0x0, TH2 *hNSigTOFNeg=0x0, TString name="", Float_t fMass=1.0, TString passName="",Float_t ySLow=0.1, Float_t ySHigh=1E4, Bool_t Print=0);




////=============== The Main Macro Starts Here: ==================


void plotSkimTreeQA2D_modified(std::string config_path="../configuration.json"){

  readConfig(config_path);

    // Construct the dataset name and path from JSON config
  // std::string cfgYear = CONFIG["dataset"]["year"].get<std::string>();
  // std::string cfgPeriod = CONFIG["dataset"]["period"].get<std::string>();
  // std::string cfgPass = CONFIG["dataset"]["pass"].get<std::string>();
  // std::string cfgDedxSelection = CONFIG["dataset"]["dEdxSelection"].get<std::string>();
  // std::string cfgHadronicRate = CONFIG["dataset"]["HadronicRate"].get<std::string>();
  // std::string cfgTag1 = CONFIG["dataset"]["optTag1"].get<std::string>();
  // std::string cfgTag2 = CONFIG["dataset"]["optTag2"].get<std::string>();
  std::string cfgSkimPath = CONFIG["dataset"]["input_skimmedtree_path"].get<std::string>();
  std::string cgfV0treename = CONFIG["general"]["V0treename"];
  std::string cgfTPCTOFtreename = CONFIG["general"]["tpctoftreename"];
  std::string dataset_name = CONFIG["output"]["general"]["name"].get<std::string>();

  // TString sDataSet = TString::Format("LHC%s%s_pass%s_%s_%s_%s_HR_%s", cfgYear.c_str(), cfgPeriod.c_str(), cfgPass.c_str(), cfgTag1.c_str(), cfgTag2.c_str(), cfgDedxSelection.c_str(), cfgHadronicRate.c_str());
  TString sDataSet = TString::Format("%s", dataset_name.c_str());
  TString path2file = TString::Format("%s", cfgSkimPath.c_str());
  gStyle->SetOptStat(0000);        //Do not draw Statistics.
  gStyle->SetImageScaling(50.);    //This seems to not work :P
  Int_t    Print = 1;              // 0 = do not save pdf. > 0 = Draw pdf figures.

  //Ad hoc local file: Comment out if you arunning in lxbk0552
  //  TFile *f1 = TFile::Open("./mergedFiles/LHC23k_newPass1/tpc_skims_merged_LHC23k.root","READ"); //sDataSet="LHC23k_NewPass1";

  // Construct the file name dynamically using sDataSet
  TString fileName = TString::Format("%s", path2file.Data());


  cout << "Looking for file: " << fileName.Data() << endl;

  // Open the file
  ifstream intxt;
  intxt.open(fileName.Data());  // Open the dynamically created file

  if (!intxt.is_open()) {
    cout << "Error: Could not open file: " << fileName.Data() << endl;
    return;
  }

  // You can now use the path2file variable to load the dataset, etc.
  TFile *f1 = TFile::Open(Form("%s", path2file.Data()), "READ");
  if (!f1 || f1->IsZombie()) {
    cout << "Error opening file: " << path2file.Data() << endl;
    return;
  }





  /// How to Load Default Parameters from CCDB:  (Thanks to Jeremy!)
  /// o2-pidparam-tpc-response --mode pull --min-runnumber < 5xxxxx >
  Double_t OptParDeft[6] = {0.280991, 3.23543, 0.0244042, 2.31595, 0.780374}; ///Default Parm LHC23

  /// Define the Bethe-Bloch Functions: // Note: this function is not used in this QA macro!!
  TF1 *funcBBvsBGDefault = new TF1("funcBBvsBGDefault","[0]*([1]- log([2]+pow(x,-1.*[4])) - pow((x/sqrt(1+x*x)),[3]))*pow(1.0,2.3)*50./pow((x/sqrt(1+x*x)),[3])",2.e-3,4.E5);
  funcBBvsBGDefault->SetParameters(OptParDeft);

  //Note: OptParDeft is the default BB Parameters used to make the Skimmed Tree. We then use our
  //BB fitting macro to get the more optimised BB parameters (by fitting the <dE/dx> vs \beta\gamma
  //with the Bethe-Bloch (ALEPH) function.



  if(f1->IsOpen()) cout<<"\nInfo:: Opening File "<<f1->GetName()<<endl;
  else{
    cout<<"\n:::WARNING::: Could not find the file, check name or path \n Quit!\n"<<endl; return;
  }


  /// Now read the branches:

  TIter keyList(f1->GetListOfKeys());
  TKey    *key;
  TTree   *fChain;
  UChar_t fPidIndex;
  Int_t   fRunNumber;
  Float_t fNSigTPC,fNSigTOF;
  Float_t fTPCSignal,fBetaGamma;
  Float_t fSigned1Pt,fTPCInnerMom;
  Float_t fInvDeDxExpTPC, fEta;
  Float_t fFt0Occ;

  TBranch *b_fEta;       //!
  TBranch *b_fNSigTPC;   //!
  TBranch *b_fNSigTOF;   //!
  TBranch *b_fPidIndex;    //!
  TBranch *b_fBetaGamma;   //!
  TBranch *b_fRunNumber;   //!
  TBranch *b_fTPCSignal;   //!
  TBranch *b_fSigned1Pt;   //!
  TBranch *b_fTPCInnerParam; //!
  TBranch *b_fInvDeDxExpTPC; //!
  TBranch *b_fFt0Occ; //!


  /// Here we are creating bins with increasing binwidth, so that in log-x axis,
  /// we see equally spaced bins:

  Double_t binEdge = 0.0;
  Double_t width = 0.01; //0.001
  const Int_t icNbinX = 200;
  Double_t profBins[icNbinX+1] = {0,};
  profBins[0]={0.005};

  cout<<"bins: ";
  for(int i=0;i<icNbinX;i++){
    width = width*exp(0.07265);
    //cout<<binEdge+width<<", ";
    profBins[i+1] = binEdge+width;
  }
  cout<<endl;


  ///Now Momentum (Pin) Bins: also with increasing binwidth
  const int nBinsPin = 40;
  Double_t profBinsPin[nBinsPin+1] = {0,};
  profBinsPin[0]={0.1};
  width=0.20;
  cout<<"bins: ";
  for(int i=0;i<nBinsPin;i++){
    //width = width*exp(0.165);
    width = width*exp(0.100183);
    cout<<width<<", ";
    profBinsPin[i+1] = width;
  }
  cout<<endl;


  //return;



  /// NsigmaTPC vs \beta\gamma
  TH2F *hNsigTPCvsBGDataPionPos = new TH2F("hNsigTPCvsBGDataPionPos","N#sigma vs BG #pi^{+}(Data)",icNbinX,profBins,200,-10,10);
  TH2F *hNsigTPCvsBGDataPionNeg = new TH2F("hNsigTPCvsBGDataPionNeg","N#sigma vs BG #pi^{-}(Data)",icNbinX,profBins,200,-10,10);
  TH2F *hNsigTPCvsBGDataKaonPos = new TH2F("hNsigTPCvsBGDataKaonPos","N#sigma vs BG  K^{+} (Data)",icNbinX,profBins,200,-10,10);
  TH2F *hNsigTPCvsBGDataKaonNeg = new TH2F("hNsigTPCvsBGDataKaonNeg","N#sigma vs BG  K^{-} (Data)",icNbinX,profBins,200,-10,10);
  TH2F *hNsigTPCvsBGDataProtPos = new TH2F("hNsigTPCvsBGDataProtPos","N#sigma vs BG  prot  (Data)",icNbinX,profBins,200,-10,10);
  TH2F *hNsigTPCvsBGDataProtNeg = new TH2F("hNsigTPCvsBGDataProtNeg","N#sigma vs BG bar{p} (Data)",icNbinX,profBins,200,-10,10);

  TH2F *hNsigTPCvsBGDataV0ElecNeg = new TH2F("hNsigTPCvsBGDataV0ElecNeg","N#sigma vs BG  e^{-}  (Data)",icNbinX,profBins,200,-10,10);
  TH2F *hNsigTPCvsBGDataV0ElecPos = new TH2F("hNsigTPCvsBGDataV0ElecPos","N#sigma vs BG  e^{+}  (Data)",icNbinX,profBins,200,-10,10);
  TH2F *hNsigTPCvsBGDataV0KaonPos = new TH2F("hNsigTPCvsBGDataV0KaonPos","N#sigma vs BG  K^{+}  (Data)",icNbinX,profBins,200,-10,10);
  TH2F *hNsigTPCvsBGDataV0KaonNeg = new TH2F("hNsigTPCvsBGDataV0KaonNeg","N#sigma vs BG  K^{-}  (Data)",icNbinX,profBins,200,-10,10);
  //left here for home... 31/10/23

  /// NsigmaTOF vs \beta\gamma
  TH2F *hNsigTOFvsBGDataPionPos = new TH2F("hNsigTOFvsBGDataPionPos","N#sigma vs BG #pi^{#pm}(Data)",icNbinX,profBins,200,-10,10);
  TH2F *hNsigTOFvsBGDataKaonPos = new TH2F("hNsigTOFvsBGDataKaonPos","N#sigma vs BG  K^{#pm} (Data)",icNbinX,profBins,200,-10,10);
  TH2F *hNsigTOFvsBGDataProtPos = new TH2F("hNsigTOFvsBGDataProtPos","N#sigma vs BG p#bar{p} (Data)",icNbinX,profBins,200,-10,10);
  TH2F *hNsigTOFvsBGDataPionNeg = new TH2F("hNsigTOFvsBGDataPionNeg","N#sigma vs BG #pi^{#pm}(Data)",icNbinX,profBins,200,-10,10);
  TH2F *hNsigTOFvsBGDataKaonNeg = new TH2F("hNsigTOFvsBGDataKaonNeg","N#sigma vs BG  K^{#pm} (Data)",icNbinX,profBins,200,-10,10);
  TH2F *hNsigTOFvsBGDataProtNeg = new TH2F("hNsigTOFvsBGDataProtNeg","N#sigma vs BG p#bar{p} (Data)",icNbinX,profBins,200,-10,10);

  TH2F *hNsigTOFvsBGDataV0ElecPos = new TH2F("hNsigTOFvsBGDataV0ElecPos","N#sigma vs BG  e^{#pm}  (Data)",icNbinX,profBins,200,-10,10);
  TH2F *hNsigTOFvsBGDataV0ElecNeg = new TH2F("hNsigTOFvsBGDataV0ElecNeg","N#sigma vs BG  e^{#pm}  (Data)",icNbinX,profBins,200,-10,10);
  TH2F *hNsigTOFvsBGDataV0KaonPos = new TH2F("hNsigTOFvsBGDataV0KaonPos","N#sigma vs BG  K^{#pm}  (Data)",icNbinX,profBins,200,-10,10);
  TH2F *hNsigTOFvsBGDataV0KaonNeg = new TH2F("hNsigTOFvsBGDataV0KaonNeg","N#sigma vs BG  K^{#pm}  (Data)",icNbinX,profBins,200,-10,10);

  ////dEdx vs Eta:
  TH2F *hdEdxvsEtaPionPosTPC = new TH2F("hdEdxvsEtaPionPosTPC","hdEdxvsEta, 1<#beta#gamma<2 #pi^{+} (TPC)",100,-1,1,200,10,210);
  TH2F *hdEdxvsEtaPionNegTPC = new TH2F("hdEdxvsEtaPionNegTPC","hdEdxvsEta, 1<#beta#gamma<2 #pi^{-} (TPC)",100,-1,1,200,10,210);
  TH2F *hdEdxvsEtaKaonPosTPC = new TH2F("hdEdxvsEtaKaonPosTPC","hdEdxvsEta, 1<#beta#gamma<2  K^{+}  (TPC)",100,-1,1,200,10,210);
  TH2F *hdEdxvsEtaKaonNegTPC = new TH2F("hdEdxvsEtaKaonNegTPC","hdEdxvsEta, 1<#beta#gamma<2  K^{-}  (TPC)",100,-1,1,200,10,210);
  TH2F *hdEdxvsEtaProtPosTPC = new TH2F("hdEdxvsEtaProtPosTPC","hdEdxvsEta, 1<#beta#gamma<2  prot   (TPC)",100,-1,1,200,10,210);
  TH2F *hdEdxvsEtaProtNegTPC = new TH2F("hdEdxvsEtaProtNegTPC","hdEdxvsEta, 1<#beta#gamma<2 #bar{p} (TPC)",100,-1,1,200,10,210);


  TH2F *hdEdxvsEtaPionPosV0 = new TH2F("hdEdxvsEtaPionPosV0","hdEdxvsEta, 1<#beta#gamma<2 #pi^{+} (V0)",100,-1,1,200,10,210);
  TH2F *hdEdxvsEtaPionNegV0 = new TH2F("hdEdxvsEtaPionNegV0","hdEdxvsEta, 1<#beta#gamma<2 #pi^{-} (V0)",100,-1,1,200,10,210);
  TH2F *hdEdxvsEtaElecPosV0 = new TH2F("hdEdxvsEtaElecPosV0","hdEdxvsEta, 1<#beta#gamma<2   e^{+} (V0)",100,-1,1,200,10,210);
  TH2F *hdEdxvsEtaElecNegV0 = new TH2F("hdEdxvsEtaElecNegV0","hdEdxvsEta, 1<#beta#gamma<2   e^{-} (V0)",100,-1,1,200,10,210);
  TH2F *hdEdxvsEtaProtPosV0 = new TH2F("hdEdxvsEtaProtPosV0","hdEdxvsEta, 1<#beta#gamma<2   prot  (V0)",100,-1,1,200,10,210);
  TH2F *hdEdxvsEtaProtNegV0 = new TH2F("hdEdxvsEtaProtNegV0","hdEdxvsEta, 1<#beta#gamma<2 #bar{p} (V0)",100,-1,1,200,10,210);
  TH2F *hdEdxvsEtaKaonPosV0 = new TH2F("hdEdxvsEtaKaonPosV0","hdEdxvsEta, 1<#beta#gamma<2  K^{+} (V0)",100,-1,1,200,10,210);
  TH2F *hdEdxvsEtaKaonNegV0 = new TH2F("hdEdxvsEtaKaonNegV0","hdEdxvsEta, 1<#beta#gamma<2  K^{-} (V0)",100,-1,1,200,10,210);


  TH2F *hNsigmaTPCvsPinProtTPC = new TH2F("hNsigmaTPCvsPinProtTPC","tpc-tof prot;P_{in} (GeV/c);N#sigma_{TPC}",nBinsPin,profBinsPin,200,-10,10);
  TH2F *hNsigmaTPCvsPinPionTPC = new TH2F("hNsigmaTPCvsPinPionTPC","tpc-tof #pi^{+};P_{in}(GeV/c);N#sigma_{TPC}",nBinsPin,profBinsPin,200,-10,10);
  TH2F *hNsigmaTPCvsPinProtV0 = new TH2F("hNsigmaTPCvsPinProtV0","V0 Prot;P_{in} (GeV/c);N#sigma_{TPC}",nBinsPin,profBinsPin,200,-10,10);
  TH2F *hNsigmaTPCvsPinPionV0 = new TH2F("hNsigmaTPCvsPinPionV0","V0 Pion;P_{in} (GeV/c);N#sigma_{TPC}",nBinsPin,profBinsPin,200,-10,10);
  TH2F *hNsigmaTPCvsPinKaonV0 = new TH2F("hNsigmaTPCvsPinKaonV0","V0 Kaon;P_{in} (GeV/c);N#sigma_{TPC}",nBinsPin,profBinsPin,200,-10,10);
  TH2F *hNsigmaTOFvsPinProtV0 = new TH2F("hNsigmaTOFvsPinProtV0","V0 Prot;P_{in} (GeV/c);N#sigma_{TOF}",nBinsPin,profBinsPin,200,-10,10);

  //return;

  TProfile *hdEdxVsBetaGammaTheo = new TProfile("hdEdxVsBetaGammaTheo","",icNbinX,profBins);



  Int_t ch = 0;
  UInt_t ipid;
  Int_t nTotTrk = 0;
  Int_t nV0Trk = 0;
  Int_t oldRun = 0, thisRun;
  Double_t dBBtheo;

  cout<<"\nReading Data: "<<sDataSet.Data()<<",\nRead Runs: ";

  //Get the keys from file and loop over them:
  while ((key = (TKey*)keyList())) {
    TClass *cl = gROOT->GetClass(key->GetClassName());
    if (!cl->InheritsFrom("TDirectoryFile")) continue;
    TString dirName = (TString ) key->GetName();

    TTree *treeTPC=NULL;
    TTree *treeV0=NULL;

    TDirectory *dir = (TDirectory*)f1->Get(Form("%s:/%s",f1->GetName(),dirName.Data()));

    dir->GetObject(cgfTPCTOFtreename.c_str(), treeTPC);

    if (treeTPC){
      fChain = treeTPC;
      fChain->SetMakeClass(1);

      fChain->SetBranchAddress("fEta", &fEta, &b_fEta);
      fChain->SetBranchAddress("fNSigTPC", &fNSigTPC, &b_fNSigTPC);
      fChain->SetBranchAddress("fNSigTOF", &fNSigTOF, &b_fNSigTOF);
      fChain->SetBranchAddress("fPidIndex", &fPidIndex, &b_fPidIndex);
      fChain->SetBranchAddress("fBetaGamma", &fBetaGamma, &b_fBetaGamma);
      fChain->SetBranchAddress("fTPCSignal", &fTPCSignal, &b_fTPCSignal);
      fChain->SetBranchAddress("fSigned1Pt", &fSigned1Pt, &b_fSigned1Pt);
      fChain->SetBranchAddress("fRunNumber", &fRunNumber, &b_fRunNumber);
      fChain->SetBranchAddress("fTPCInnerParam", &fTPCInnerMom, &b_fTPCInnerParam);
      fChain->SetBranchAddress("fInvDeDxExpTPC", &fInvDeDxExpTPC, &b_fInvDeDxExpTPC);
      fChain->SetBranchAddress("fFt0Occ", &fFt0Occ, &b_fFt0Occ);

      Long64_t nTrk = fChain->GetEntries();
      cout<<"Reading directory: "<<dirName.Data()<<" No. of total tracks: "<<nTrk<<endl;

      for (Int_t i = 0; i < nTrk; i++) {
	fChain->GetEntry(i);

	dBBtheo = funcBBvsBGDefault->Eval(fBetaGamma); /// not used in this macro

	//if(nTotTrk>6E6) break;           //if we want to test our macro for small sample of tracks!
	//if(fRunNumber==528296) continue; //if we want to skip some specific run

	ipid = unsigned(fPidIndex);
	if(fSigned1Pt>0) ch=1;
	else ch=-1;

	if(oldRun!=fRunNumber){
	  oldRun = fRunNumber;
	  cout<<oldRun<<", ";
	}
	//check the dEdx vs eta for a small \beta\Gamma range: (pT integrated)
	if(fBetaGamma>1 && fBetaGamma<2){
	  if(fabs(ipid)==2){
	    if(ch>0)
	      hdEdxvsEtaPionPosTPC->Fill(fEta,fTPCSignal);
	    else
	      hdEdxvsEtaPionNegTPC->Fill(fEta,fTPCSignal);
	  }
	  else if(fabs(ipid==3)){
	    if(ch>0)
	      hdEdxvsEtaKaonPosTPC->Fill(fEta,fTPCSignal);
	    else
	      hdEdxvsEtaKaonNegTPC->Fill(fEta,fTPCSignal);
	  }
	  else if(fabs(ipid==4)){
	    if(ch>0)
	      hdEdxvsEtaProtPosTPC->Fill(fEta,fTPCSignal);
	    else
	      hdEdxvsEtaProtNegTPC->Fill(fEta,fTPCSignal);
	  }
	}

	///fill the TPC nSigma: (eta and pt integrated)
	if(ch>0){
	  if(fabs(ipid)==2){
	    hNsigTPCvsBGDataPionPos->Fill(fBetaGamma,fNSigTPC);
	    hNsigTOFvsBGDataPionPos->Fill(fBetaGamma,fNSigTOF);
	    hNsigmaTPCvsPinPionTPC->Fill(fTPCInnerMom,fNSigTPC);
	  }
	  else if(fabs(ipid)==3){
	    hNsigTPCvsBGDataKaonPos->Fill(fBetaGamma,fNSigTPC);
	    hNsigTOFvsBGDataKaonPos->Fill(fBetaGamma,fNSigTOF);
	  }
	  else if(fabs(ipid)==4){
	    hNsigTPCvsBGDataProtPos->Fill(fBetaGamma,fNSigTPC);
	    hNsigTOFvsBGDataProtPos->Fill(fBetaGamma,fNSigTOF);
	    hNsigmaTPCvsPinProtTPC->Fill(fTPCInnerMom,fNSigTPC);
	  }
	}
	else{
	  if(fabs(ipid)==2){
	     hNsigTPCvsBGDataPionNeg->Fill(fBetaGamma,fNSigTPC);
	     hNsigTOFvsBGDataPionNeg->Fill(fBetaGamma,fNSigTOF);
	  }
	  else if(fabs(ipid)==3){
	    hNsigTPCvsBGDataKaonNeg->Fill(fBetaGamma,fNSigTPC);
	    hNsigTOFvsBGDataKaonNeg->Fill(fBetaGamma,fNSigTOF);
	  }
	  else if(fabs(ipid)==4){
	    hNsigTPCvsBGDataProtNeg->Fill(fBetaGamma,fNSigTPC);
	    hNsigTOFvsBGDataProtNeg->Fill(fBetaGamma,fNSigTOF);
	  }
	}
	//hdEdxVsBetaGammaTheo->Fill(fBetaGamma,1./fInvDeDxExpTPC);
	nTotTrk++;
      }//track loop
    }//directory loop




    dir->GetObject(cgfV0treename.c_str(),treeV0);
    if(treeV0){
      //cout<<" Found the V0 Tree "<<endl;
      fChain = treeV0;
      fChain->SetMakeClass(1);
      fChain->SetBranchAddress("fEta", &fEta, &b_fEta);
      fChain->SetBranchAddress("fNSigTPC", &fNSigTPC, &b_fNSigTPC);
      fChain->SetBranchAddress("fNSigTOF", &fNSigTOF, &b_fNSigTOF);
      fChain->SetBranchAddress("fPidIndex", &fPidIndex, &b_fPidIndex);
      fChain->SetBranchAddress("fTPCSignal", &fTPCSignal, &b_fTPCSignal);
      fChain->SetBranchAddress("fBetaGamma", &fBetaGamma, &b_fBetaGamma);
      fChain->SetBranchAddress("fSigned1Pt", &fSigned1Pt, &b_fSigned1Pt);
      fChain->SetBranchAddress("fRunNumber", &fRunNumber, &b_fRunNumber);
      fChain->SetBranchAddress("fTPCInnerParam", &fTPCInnerMom, &b_fTPCInnerParam);
      fChain->SetBranchAddress("fInvDeDxExpTPC", &fInvDeDxExpTPC, &b_fInvDeDxExpTPC);
      fChain->SetBranchAddress("fFt0Occ", &fFt0Occ, &b_fFt0Occ);

      Long64_t nTrk = fChain->GetEntries();
      cout<<"Reading directory: "<<dirName.Data()<<" No. of total tracks: "<<nTrk<<endl;

      for (Int_t i = 0; i < nTrk; i++) {
	fChain->GetEntry(i);

	//if(nV0Trk>2E6) break;
	//if(fRunNumber==528296) continue; //if we want to skip some specific run

	if(fSigned1Pt>0) ch=1;
	else ch=-1;

	ipid = unsigned(fPidIndex);

	if(fBetaGamma>1E3 && fBetaGamma<1E4){
	  if(fabs(ipid)==0){
	    if(ch>0)
	      hdEdxvsEtaElecPosV0->Fill(fEta,fTPCSignal);
	    else
	      hdEdxvsEtaElecNegV0->Fill(fEta,fTPCSignal);
	  }
    if(fBetaGamma>1 && fBetaGamma<2){
      if(fabs(ipid)==3){
        if(ch>0) hdEdxvsEtaKaonPosV0->Fill(fEta,fTPCSignal);
        else hdEdxvsEtaKaonNegV0->Fill(fEta,fTPCSignal);
      }
    }
	}
	if(ch>0){
	  if(fabs(ipid)==0 && fabs(fNSigTPC)<4.0){
	    hNsigTPCvsBGDataV0ElecPos->Fill(fBetaGamma,fNSigTPC);
	    hNsigTOFvsBGDataV0ElecPos->Fill(fBetaGamma,fNSigTOF);
	  }
	  else if(fabs(ipid)==2 && fabs(fNSigTPC)<4.0){
	    //hdEdxvsMomDataV0Pion->Fill(fBetaGamma,fTPCSignal);
	    hNsigmaTPCvsPinPionV0->Fill(fTPCInnerMom,fNSigTPC);
	  }
    else if(fabs(ipid)==3 && fabs(fNSigTPC)<4.0){
      hNsigTPCvsBGDataV0KaonPos->Fill(fBetaGamma,fNSigTPC);
      hNsigTOFvsBGDataV0KaonPos->Fill(fBetaGamma,fNSigTOF);
      hNsigmaTPCvsPinKaonV0->Fill(fTPCInnerMom,fNSigTPC);
    }
	  else if(fabs(ipid)==4){
	    //hdEdxvsMomDataV0Prot->Fill(fBetaGamma,fTPCSignal);
	    hNsigmaTPCvsPinProtV0->Fill(fTPCInnerMom,fNSigTPC);
	    hNsigmaTOFvsPinProtV0->Fill(fTPCInnerMom,fNSigTOF);
	  }
	}
	else{
	  if(fabs(ipid)==0 && fabs(fNSigTPC)<4.0){
	    //hdEdxvsMomDataV0ElecNeg->Fill(fBetaGamma,fTPCSignal);
	    hNsigTPCvsBGDataV0ElecNeg->Fill(fBetaGamma,fNSigTPC);
	    hNsigTOFvsBGDataV0ElecNeg->Fill(fBetaGamma,fNSigTOF);
	  }
	  else if(fabs(ipid)==2 && fabs(fNSigTPC)<4.0){
	    //hdEdxvsMomDataV0PionNeg->Fill(fBetaGamma,fTPCSignal);
	  }
    else if(fabs(ipid)==3 && fabs(fNSigTPC)<4.0){
      hNsigTPCvsBGDataV0KaonNeg->Fill(fBetaGamma,fNSigTPC);
      hNsigTOFvsBGDataV0KaonNeg->Fill(fBetaGamma,fNSigTOF);
    }
	  else if(fabs(ipid)==4){
	    //hdEdxvsMomDataV0ProtNeg->Fill(fBetaGamma,fTPCSignal);

	  }
	}
	nV0Trk++;
      }
    }


  }//While loop

  cout<<endl;

  /// For test only:
  /* TCanvas *c1,*c2,*c3,*c4;
  c1 = new TCanvas("c1","",40,40,500,360);
  hNsigmaTPCvsPinProtV0->Draw("COLZ");
  c2 = new TCanvas("c2","",140,40,500,360);
  hNsigmaTPCvsPinProtTPC->Draw("COLZ");
  c3 = new TCanvas("c3","",240,40,500,360);
  hNsigmaTPCvsPinPionTPC->Draw("COLZ"); return; */


  hdEdxvsEtaPionPosTPC->RebinX(2);
  hdEdxvsEtaPionNegTPC->RebinX(2);
  hdEdxvsEtaKaonPosTPC->RebinX(2);
  hdEdxvsEtaKaonNegTPC->RebinX(2);
  hdEdxvsEtaProtPosTPC->RebinX(2);
  hdEdxvsEtaProtNegTPC->RebinX(2);
  hdEdxvsEtaElecPosV0->RebinX(2);
  hdEdxvsEtaElecNegV0->RebinX(2);
  hdEdxvsEtaKaonPosV0->RebinX(2);
  hdEdxvsEtaKaonNegV0->RebinX(2);

  plotdEdxvsEta(hdEdxvsEtaPionPosTPC, hdEdxvsEtaPionNegTPC,"Pion_TPC", 0.140, sDataSet, 38, 124, Print);
  xpos+=100;
  plotdEdxvsEta(hdEdxvsEtaKaonPosTPC, hdEdxvsEtaKaonNegTPC,"Kaon_TPC", 0.140, sDataSet, 38, 124, Print);
  xpos+=100;
  plotdEdxvsEta(hdEdxvsEtaProtPosTPC, hdEdxvsEtaProtNegTPC,"Prot_TPC", 0.140, sDataSet, 38, 124, Print);
  xpos+=100;
  plotdEdxvsEta(hdEdxvsEtaElecPosV0, hdEdxvsEtaElecNegV0, "Elec_V0", 0.140, sDataSet, 38, 124, Print);
  xpos+=100;
  plotdEdxvsEta(hdEdxvsEtaKaonPosV0, hdEdxvsEtaKaonNegV0, "Kaon_V0", 0.140, sDataSet, 38, 124, Print);


  //return;

  xpos+=100;
  plotNsigmaVsBG(hNsigTPCvsBGDataPionPos, hNsigTPCvsBGDataPionNeg, hNsigTOFvsBGDataPionPos, hNsigTOFvsBGDataPionNeg, "Pion_TPCToF", 0.140, sDataSet, -3.6, 4.2, Print);
  xpos+=100;
  plotNsigmaVsBG(hNsigTPCvsBGDataKaonPos, hNsigTPCvsBGDataKaonNeg, hNsigTOFvsBGDataKaonPos, hNsigTOFvsBGDataKaonNeg, "Kaon_TPCToF", 0.495, sDataSet, -3.6, 4.2, Print);
  xpos+=100;
  plotNsigmaVsBG(hNsigTPCvsBGDataProtPos, hNsigTPCvsBGDataProtNeg, hNsigTOFvsBGDataProtPos, hNsigTOFvsBGDataProtNeg, "Prot_TPCToF", 0.938, sDataSet, -3.6, 4.2, Print);
  xpos+=100;
  plotNsigmaVsBG(hNsigTPCvsBGDataV0ElecPos, hNsigTPCvsBGDataV0ElecNeg, hNsigTOFvsBGDataV0ElecPos, hNsigTOFvsBGDataV0ElecNeg,"Elec_V0",0.511E-3,sDataSet,-3.6,4.2, Print);
  xpos+=100;
  plotNsigmaVsBG(hNsigTPCvsBGDataV0KaonPos, hNsigTPCvsBGDataV0KaonNeg, hNsigTOFvsBGDataV0KaonPos, hNsigTOFvsBGDataV0KaonNeg,"Kaon_V0",0.495,sDataSet,-3.6,4.2, Print);


  //plotNsigmavsPin
  ypos+=100;
  xpos =50;
  plotNsigmaVsPin(hNsigmaTPCvsPinProtV0, hNsigmaTPCvsPinProtTPC, hNsigmaTPCvsPinPionV0, hNsigmaTPCvsPinPionTPC, "Pion_Prot", 0.140, sDataSet, -3.6, 3.4, Print);
  xpos+=100;
  plotNsigmaVsPin(hNsigmaTPCvsPinKaonV0, hNsigmaTPCvsPinPionTPC, hNsigmaTPCvsPinKaonV0, hNsigmaTPCvsPinKaonV0, "Kaon_V0", 0.495, sDataSet, -3.6, 3.4, Print);




}
//========== main ends ==============

///////////////////////////////////////////////
//   The place below is for Functions Only   //
///////////////////////////////////////////////


TGraphErrors* getGraphMeanYFrom2DHist(TH2 *hNSigTPCpos=0x0,Float_t yLow=-1E2,Float_t yHigh=1E3){

  const char *histName = hNSigTPCpos->GetName();

  if(fabs(yLow)>0 && fabs(yHigh)>0){
    hNSigTPCpos->GetYaxis()->SetRangeUser(yLow,yHigh);
  }

  hNSigTPCpos->FitSlicesY();
  TH1D *hMeansHist = (TH1D*) gDirectory->Get(Form("%s_1",histName));
  TH1D *hSigmaHist = (TH1D*) gDirectory->Get(Form("%s_2",histName));
  //hMeansHist->Draw(); //Debug
  TGraphErrors *graphMean = new TGraphErrors();
  graphMean->SetName(histName); graphMean->SetTitle(histName);


  int nbinx = hMeansHist->GetNbinsX();
  double mean,sigma,xcent;
  int ip = 0;
  for(int i=1; i<=nbinx; nbinx++){
    mean  = hMeansHist->GetBinContent(i);
    xcent = hMeansHist->GetBinCenter(i);
    //if(fabs(mean)!=0)
    graphMean->SetPoint(ip,xcent,mean);
    ip++;
  }

  return graphMean;
}

void plotdEdxvsEta(TH2 *hNSigTPCpos=0x0, TH2 *hNSigTPCneg=0x0, TString name="", Float_t fMass=1.0, TString passName="",Float_t ySLow, Float_t ySHigh, Bool_t Print){
  TLine   *line;
  TLatex  *tex;
  TLegend *legend;
  Double_t yLblPos = 0.825, xLblPos = 0.85;
  Double_t xSLow = -0.81, xSHigh=0.81;

  //--------- Do the slice fitting ------

  const char *histNamePos = hNSigTPCpos->GetName();
  hNSigTPCpos->GetYaxis()->SetRangeUser(50,100);
  hNSigTPCpos->FitSlicesY();
  TH1D *hMeansHistPos = (TH1D*) gDirectory->Get(Form("%s_1",histNamePos));
  TH1D *hSigmaHistPos = (TH1D*) gDirectory->Get(Form("%s_2",histNamePos));

  const char *histNameNeg = hNSigTPCneg->GetName();
  hNSigTPCneg->GetYaxis()->SetRangeUser(50,100);
  hNSigTPCneg->FitSlicesY();
  TH1D *hMeansHistNeg = (TH1D*) gDirectory->Get(Form("%s_1",histNameNeg));
  TH1D *hSigmaHistNeg = (TH1D*) gDirectory->Get(Form("%s_2",histNameNeg));

  cout<<"Done the Slice fits for "<<histNamePos<<" and "<<histNameNeg<<endl;

  //hMeansHist->Draw(); //Debug
  TGraphErrors *graphMeanPos = new TGraphErrors();
  graphMeanPos->SetName(histNamePos);
  graphMeanPos->SetTitle(histNamePos);
  TGraphErrors *graphMeanNeg = new TGraphErrors();
  graphMeanNeg->SetName(histNameNeg);
  graphMeanNeg->SetTitle(histNameNeg);


  //TGraphErrors *graphMeanPos = (TGraphErrors *) getGraphMeanYFrom2DHist(hNSigTPCpos,50.,100.);  // error: call to 'getGraphMeanYFrom2DHist' is ambiguous ???
  //TGraphErrors *graphMeanNeg = (TGraphErrors *) getGraphMeanYFrom2DHist(hNSigTPCneg,50.,100.);


  int nbinx = hMeansHistPos->GetNbinsX();
  double mean,sigma,xcent;
  int ip = 0;
  for(int i=1; i<=nbinx; i++){

    mean  = hMeansHistPos->GetBinContent(i);
    xcent = hMeansHistPos->GetBinCenter(i);
    //if(fabs(mean)!=0)
    graphMeanPos->SetPoint(ip,xcent,mean);
    //cout<<"bin "<<i<<" eta: "<<xcent<<" mean: "<<mean<<endl;
    mean  = hMeansHistNeg->GetBinContent(i);
    xcent = hMeansHistNeg->GetBinCenter(i);
    graphMeanNeg->SetPoint(ip,xcent,mean);
    ip++;
  }
  //-------------------------------------
  //cout<<"\n Filled Graphs, Now Drawing..."<<endl;


  TCanvas *Canvas = GetCanvas(Form("CandEdxvsEta%s",name.Data()),xpos,ypos,cansizeX,cansizeY,0,0,0.02,0.02,0.14,0.01);
  Canvas->cd();
  //Func: GetPad(name.Data(), xpos1, ypos1, xpos2, ypos2, topMar, botMar, leftMar, rightMar)
  TPad *padTop = GetPad(Form("padTop%s",name.Data()),0.0,0.525,1.0,1.0, 0.02,0.0,0.14,0.10);
  padTop->Draw();
  padTop->cd();
  padTop->SetTicks();
  //padTop->SetLogx();
  padTop->SetLogz();
  padTop->SetGridx();
  padTop->SetGridy();
  hNSigTPCpos->SetTitle("");
  SetTitleTH1(hNSigTPCpos,"dE/dx (+Ve)",0.065,0.85,"#eta",0.06,xLblPos);
  SetAxisTH1(hNSigTPCpos,ySLow,ySHigh,xSLow,xSHigh,0.055,0.055,0.055,-0.01);
  hNSigTPCpos->Draw("COLZ");
  SetMarkerTH1(graphMeanPos,"",25,0.5,2,2);
  graphMeanPos->Draw("PSAME");
  SetMarkerTH1(graphMeanNeg,"",24,0.5,4,4);
  graphMeanNeg->Draw("PSAME");

  legend = new TLegend(0.55,0.76,0.85,0.95);
  legend->SetBorderSize(0);
  legend->SetFillColor(0);
  legend->SetTextFont(42);
  legend->SetTextSize(0.06);
  legend->AddEntry(hNSigTPCpos,Form("%s,",passName.Data()),"");
  legend->AddEntry(hNSigTPCpos,Form("%s",name.Data()),"");
  if(!strncmp(name,"Elec_V0",4)){
    legend->AddEntry(hNSigTPCpos,"10^{3} #leq #beta#gamma #leq 10^{4}","");
  }
  else
    legend->AddEntry(hNSigTPCpos,"1.0 #leq #beta#gamma #leq 2.0","");
  legend->Draw();


  Canvas->cd();
  TPad *pad2nd = GetPad(Form("pad2nd%s",name.Data()),0.0,0.00,1.0,0.525, 0.02,0.14,0.14,0.10);
  pad2nd->Draw();
  pad2nd->cd();
  pad2nd->SetTicks();
  //pad2nd->SetLogx();
  pad2nd->SetLogz();
  pad2nd->SetGridx();
  pad2nd->SetGridy();
  hNSigTPCneg->SetTitle("");
  SetTitleTH1(hNSigTPCneg,"dE/dx  (-Ve)",0.06,0.85,"#eta",0.06,xLblPos);
  SetAxisTH1(hNSigTPCneg,ySLow,ySHigh,xSLow,xSHigh,0.05,0.05,0.05,-0.01);
  hNSigTPCneg->Draw("COLZ");
  graphMeanNeg->Draw("PSAME");

  Canvas->Update() ;

  if(Print)
    {
      std::string qaPath = CONFIG["output"]["skimTreeQA"]["QApath"].get<std::string>();
      TString qaPathT = TString::Format("%s", qaPath.c_str());
      if (!qaPathT.EndsWith("/")) qaPathT += "/";
      Canvas->SaveAs(Form("%sdEdxvsEta_%s%s.pdf", qaPathT.Data(), name.Data(), passName.Data()));
    }

}






void plotNsigmaVsPin(TH2 *hNSigTPCPos=0x0, TH2 *hNSigTPCNeg=0x0, TH2 *hNSigTOFPos=0x0, TH2 *hNSigTOFNeg=0x0, TString name="", Float_t fMass=1.0, TString passName="",Float_t ySLow, Float_t ySHigh, Bool_t Print){

  TLine *line;
  TLatex  *tex;
  TLegend *legend;
  Double_t yLblPos = 0.825, xLblPos = 0.85;
  Double_t xSLow = 0.11, xSHigh=105;
  ///Different Range for Electron:

  if(!strncmp(name,"Elec_V0",4)){
    xSLow = 140;
    xSHigh=2E4;
  }
  if(!strncmp(name,"Pion",4)){
    xSLow=0.44;
    xSHigh=105;
  }



  //--------- Do the slice fitting ------
  const char *histNamePos = hNSigTPCPos->GetName();
  hNSigTPCPos->GetYaxis()->SetRangeUser(-3.0,3.0);
  hNSigTPCPos->FitSlicesY();
  TH1D *hMeansHistPos = (TH1D*) gDirectory->Get(Form("%s_1",histNamePos));
  TH1D *hSigmaHistPos = (TH1D*) gDirectory->Get(Form("%s_2",histNamePos));

  const char *histNamePosTOF = hNSigTOFPos->GetName();
  hNSigTOFPos->GetYaxis()->SetRangeUser(-3.0,3.0);
  hNSigTOFPos->FitSlicesY();
  TH1D *hMeansHistPosTOF = (TH1D*) gDirectory->Get(Form("%s_1",histNamePosTOF));
  TH1D *hSigmaHistPosTOF = (TH1D*) gDirectory->Get(Form("%s_2",histNamePosTOF));

  const char *histNameNeg = hNSigTPCNeg->GetName();
  hNSigTPCNeg->GetYaxis()->SetRangeUser(-3.0,3.0);
  hNSigTPCNeg->FitSlicesY();
  TH1D *hMeansHistNeg = (TH1D*) gDirectory->Get(Form("%s_1",histNameNeg));
  TH1D *hSigmaHistNeg = (TH1D*) gDirectory->Get(Form("%s_2",histNameNeg));

  const char *histNameNegTOF = hNSigTOFNeg->GetName();
  hNSigTOFNeg->GetYaxis()->SetRangeUser(-3.0,3.0);
  hNSigTOFNeg->FitSlicesY();
  TH1D *hMeansHistNegTOF = (TH1D*) gDirectory->Get(Form("%s_1",histNameNegTOF));
  TH1D *hSigmaHistNegTOF = (TH1D*) gDirectory->Get(Form("%s_2",histNameNegTOF));



  cout<<"Done the Slice fits for "<<histNamePos<<" and "<<histNameNeg<<endl;

  //hMeansHist->Draw(); //Debug
  TGraphErrors *graphMeanTPCPos = new TGraphErrors();
  graphMeanTPCPos->SetName(histNamePos);
  graphMeanTPCPos->SetTitle(histNamePos);
  TGraphErrors *graphMeanTPCNeg = new TGraphErrors();
  graphMeanTPCNeg->SetName(histNameNeg);
  graphMeanTPCNeg->SetTitle(histNameNeg);
  TGraphErrors *graphMeanTOFPos = new TGraphErrors();
  graphMeanTOFPos->SetName(histNamePosTOF);
  graphMeanTOFPos->SetTitle(histNamePosTOF);
  TGraphErrors *graphMeanTOFNeg = new TGraphErrors();
  graphMeanTOFNeg->SetName(histNameNegTOF);
  graphMeanTOFNeg->SetTitle(histNameNegTOF);

  int nbinx = hMeansHistPos->GetNbinsX();
  double mean,sigma,xcent;
  int ipos = 0, ineg=0;
  int iTOFpos = 0, iTOFneg=0;

  for(int i=1; i<=nbinx; i++){
    //Fill TPC means:
    mean  = hMeansHistPos->GetBinContent(i);
    xcent = hMeansHistPos->GetBinCenter(i);
    if(fabs(mean)!=0){
      graphMeanTPCPos->SetPoint(ipos,xcent,mean);
      ipos++;
    }
    //cout<<"bin "<<i<<" eta: "<<xcent<<" mean: "<<mean<<endl;
    mean  = hMeansHistNeg->GetBinContent(i);
    xcent = hMeansHistNeg->GetBinCenter(i);
    if(fabs(mean)!=0){
      graphMeanTPCNeg->SetPoint(ineg,xcent,mean);
      ineg++;
    }
  }

  nbinx = hMeansHistPosTOF->GetNbinsX();
  for(int i=1; i<=nbinx; i++){
    ///Fill TOF means:
    mean  = hMeansHistPosTOF->GetBinContent(i);
    xcent = hMeansHistPosTOF->GetBinCenter(i);
    if(fabs(mean)!=0){
      graphMeanTOFPos->SetPoint(iTOFpos,xcent,mean);
      iTOFpos++;
    }
    //cout<<"bin "<<i<<" eta: "<<xcent<<" mean: "<<mean<<endl;
    mean  = hMeansHistNegTOF->GetBinContent(i);
    xcent = hMeansHistNegTOF->GetBinCenter(i);
    if(fabs(mean)!=0){
      graphMeanTOFNeg->SetPoint(iTOFneg,xcent,mean);
      iTOFneg++;
    }
  }
  //-------------------------------------



  TCanvas *Canvas = GetCanvas(Form("CNsimavsBG%s",name.Data()),xpos,ypos,cansizeX,cansizeY,1,1,0.02,0.02,0.14,0.01);
  Canvas->cd();
  //Func: GetPad(name.Data(), xpos1, ypos1, xpos2, ypos2, topMar, botMar, leftMar, rightMar)
  TPad *padTop = GetPad(Form("padTop%s",name.Data()),0.0,0.771,1.0,1.0, 0.02,0.0,0.14,0.10);
  padTop->Draw();
  padTop->cd();
  padTop->SetTicks();
  padTop->SetGrid();
  //padTop->SetLogx();
  padTop->SetLogz();
  padTop->SetGridy();
  hNSigTPCPos->SetTitle("");
  SetTitleTH1(hNSigTPCPos,"N#sigma_{TPC} V0 p",0.12,0.5,"P_{in} (GeV/c)",0.08,xLblPos);
  SetAxisTH1(hNSigTPCPos,ySLow,ySHigh,xSLow,xSHigh,0.12,0.1,0.085,-0.01);
  hNSigTPCPos->Draw("COLZ");
  hNSigTPCPos->GetXaxis()->SetNdivisions(512);

  SetMarkerTH1(graphMeanTPCPos,"",25,0.5,2,2);
  graphMeanTPCPos->Draw("PSAME");
  SetMarkerTH1(graphMeanTPCNeg,"",24,0.5,4,4);
  graphMeanTPCNeg->Draw("PSAME");
  //------------------------------------------------------


  legend = new TLegend(0.65,0.75,0.875,0.95);
  legend->SetBorderSize(0);
  legend->SetFillColor(0);
  legend->SetTextFont(42);
  legend->SetTextSize(0.10);
  legend->AddEntry(hNSigTPCPos,Form("%s,",passName.Data()),"");
  legend->AddEntry(hNSigTPCPos,Form("%s",name.Data()),"");
  legend->Draw();


  Canvas->cd();
  //Func:  GetPad(name.Data(), xpos1, ypos1, xpos2, ypos2, topMar, botMar, leftMar, rightMar)
  TPad *pad2nd = GetPad(Form("pad2nd%s",name.Data()),0.0,0.521,1.0,0.771, 0.02,0.00,0.14,0.10);
  pad2nd->Draw();
  pad2nd->cd();
  pad2nd->SetTicks();
  pad2nd->SetGrid();
  //pad2nd->SetLogx();
  pad2nd->SetLogz();
  pad2nd->SetGridy();
  hNSigTPCNeg->SetTitle("");
  SetTitleTH1(hNSigTPCNeg,"N#sigma_{TPC}, tpc-tof p",0.12,0.5,"P_{in} (GeV/c)",0.08,xLblPos);
  SetAxisTH1(hNSigTPCNeg,ySLow,ySHigh,xSLow,xSHigh,0.12,0.1,0.085,-0.01);
  hNSigTPCNeg->Draw("COLZ");
  hNSigTPCNeg->GetXaxis()->SetNdivisions(512);
  SetMarkerTH1(graphMeanTPCNeg,"",24,0.5,4,4);
  graphMeanTPCNeg->Draw("PSAME");


  Canvas->cd();
  TPad *pad3rd = GetPad(Form("pad3rd%s",name.Data()),0.0,0.276,1.0,0.521, 0.02,0.0,0.14,0.10);
  pad3rd->Draw();
  pad3rd->cd();
  pad3rd->SetTicks();
  pad3rd->SetGrid();
  //pad3rd->SetLogx();
  pad3rd->SetLogz();
  pad3rd->SetGridy();
  hNSigTOFPos->SetTitle("");
  SetTitleTH1(hNSigTOFPos,"N#sigma_{TPC} V0 #pi^{+}",0.12,0.5,"P_{in} (GeV/c)",0.06,xLblPos);
  SetAxisTH1(hNSigTOFPos,ySLow,ySHigh,xSLow,xSHigh,0.12,0.1,0.085,-0.01);
  hNSigTOFPos->Draw("COLZ");
  hNSigTOFPos->GetXaxis()->SetNdivisions(512);

  SetMarkerTH1(graphMeanTOFPos,"",25,0.5,2,2);
  graphMeanTOFPos->Draw("PSAME");
  SetMarkerTH1(graphMeanTOFNeg,"",24,0.5,4,4);
  graphMeanTOFNeg->Draw("PSAME");


  Canvas->cd();
  TPad *pad4th = GetPad(Form("pad4th%s",name.Data()),0.0,0.0,1.0,0.276, 0.02,0.20,0.14,0.10);
  pad4th->Draw();
  pad4th->cd();
  pad4th->SetTicks();
  pad4th->SetGrid();
  //pad4th->SetLogx();
  pad4th->SetLogz();
  pad4th->SetGridy();
  hNSigTOFNeg->SetTitle("");
  SetTitleTH1(hNSigTOFNeg,"N#sigma_{TPC} tpc-tof #pi^{+}",0.11,0.5,"P_{in} (GeV/c)",0.10,xLblPos);
  SetAxisTH1(hNSigTOFNeg,ySLow,ySHigh,xSLow,xSHigh,0.11,0.1,0.085,-0.01);
  hNSigTOFNeg->Draw("COLZ");
  hNSigTOFNeg->GetXaxis()->SetNdivisions(512);

  graphMeanTOFNeg->Draw("PSAME");


  Canvas->Update();
  if(Print)
    {
      std::string qaPath = CONFIG["output"]["skimTreeQA"]["QApath"].get<std::string>();
      TString qaPathT = TString::Format("%s", qaPath.c_str());
      if (!qaPathT.EndsWith("/")) qaPathT += "/";
      Canvas->SaveAs(Form("%sNsigmavsMomentumPin_%s%s.pdf", qaPathT.Data(), name.Data(), passName.Data()));
    }
}

///-------- The above function is for nSigma vs Pin......





//------- The function below is for nSigma  vs BetaGamma......
void plotNsigmaVsBG(TH2 *hNSigTPCPos=0x0, TH2 *hNSigTPCNeg=0x0, TH2 *hNSigTOFPos=0x0, TH2 *hNSigTOFNeg=0x0, TString name="", Float_t fMass=1.0, TString passName="",Float_t ySLow, Float_t ySHigh, Bool_t Print){

  TLine *line;
  TLatex  *tex;
  TLegend *legend;
  Double_t yLblPos = 0.825, xLblPos = 0.85;
  Double_t xSLow = 0.11, xSHigh=105;
  ///Different Range for Electron:

  if(!strncmp(name,"Elec_V0",4)){
    xSLow = 140;
    xSHigh=2E4;
  }
  if(!strncmp(name,"Pion",4)){
    xSLow=0.44;
    xSHigh=105;
  }



  //--------- Do the slice fitting ------
  const char *histNamePos = hNSigTPCPos->GetName();
  hNSigTPCPos->GetYaxis()->SetRangeUser(-3.0,3.0);
  hNSigTPCPos->FitSlicesY();
  TH1D *hMeansHistPos = (TH1D*) gDirectory->Get(Form("%s_1",histNamePos));
  TH1D *hSigmaHistPos = (TH1D*) gDirectory->Get(Form("%s_2",histNamePos));

  const char *histNamePosTOF = hNSigTOFPos->GetName();
  hNSigTOFPos->GetYaxis()->SetRangeUser(-3.0,3.0);
  hNSigTOFPos->FitSlicesY();
  TH1D *hMeansHistPosTOF = (TH1D*) gDirectory->Get(Form("%s_1",histNamePosTOF));
  TH1D *hSigmaHistPosTOF = (TH1D*) gDirectory->Get(Form("%s_2",histNamePosTOF));

  //TCanvas *c1 = new TCanvas(); c1->cd(); hMeansHistPos->Draw("P"); hMeansHistPosTOF->SetMarkerColor(2);hMeansHistPosTOF->Draw("PSAME"); return;

  const char *histNameNeg = hNSigTPCNeg->GetName();
  hNSigTPCNeg->GetYaxis()->SetRangeUser(-3.0,3.0);
  hNSigTPCNeg->FitSlicesY();
  TH1D *hMeansHistNeg = (TH1D*) gDirectory->Get(Form("%s_1",histNameNeg));
  TH1D *hSigmaHistNeg = (TH1D*) gDirectory->Get(Form("%s_2",histNameNeg));

  const char *histNameNegTOF = hNSigTOFNeg->GetName();
  hNSigTOFNeg->GetYaxis()->SetRangeUser(-3.0,3.0);
  hNSigTOFNeg->FitSlicesY();
  TH1D *hMeansHistNegTOF = (TH1D*) gDirectory->Get(Form("%s_1",histNameNegTOF));
  TH1D *hSigmaHistNegTOF = (TH1D*) gDirectory->Get(Form("%s_2",histNameNegTOF));



  cout<<"Done the Slice fits for "<<histNamePos<<" and "<<histNameNeg<<endl;

  //hMeansHist->Draw(); //Debug
  TGraphErrors *graphMeanTPCPos = new TGraphErrors();
  graphMeanTPCPos->SetName(histNamePos);
  graphMeanTPCPos->SetTitle(histNamePos);
  TGraphErrors *graphMeanTPCNeg = new TGraphErrors();
  graphMeanTPCNeg->SetName(histNameNeg);
  graphMeanTPCNeg->SetTitle(histNameNeg);
  TGraphErrors *graphMeanTOFPos = new TGraphErrors();
  graphMeanTOFPos->SetName(histNamePosTOF);
  graphMeanTOFPos->SetTitle(histNamePosTOF);
  TGraphErrors *graphMeanTOFNeg = new TGraphErrors();
  graphMeanTOFNeg->SetName(histNameNegTOF);
  graphMeanTOFNeg->SetTitle(histNameNegTOF);

  int nbinx = hMeansHistPos->GetNbinsX();
  double mean,sigma,xcent;
  int ipos = 0, ineg=0;
  int iTOFpos = 0, iTOFneg=0;

  for(int i=1; i<=nbinx; i++){
    //Fill TPC means:
    mean  = hMeansHistPos->GetBinContent(i);
    xcent = hMeansHistPos->GetBinCenter(i);
    if(fabs(mean)!=0){
      graphMeanTPCPos->SetPoint(ipos,xcent,mean);
      ipos++;
    }
    //cout<<"bin "<<i<<" eta: "<<xcent<<" mean: "<<mean<<endl;
    mean  = hMeansHistNeg->GetBinContent(i);
    xcent = hMeansHistNeg->GetBinCenter(i);
    if(fabs(mean)!=0){
      graphMeanTPCNeg->SetPoint(ineg,xcent,mean);
      ineg++;
    }
  }

  nbinx = hMeansHistPosTOF->GetNbinsX();
  for(int i=1; i<=nbinx; i++){
    ///Fill TOF means:
    mean  = hMeansHistPosTOF->GetBinContent(i);
    xcent = hMeansHistPosTOF->GetBinCenter(i);
    if(fabs(mean)!=0){
      graphMeanTOFPos->SetPoint(iTOFpos,xcent,mean);
      iTOFpos++;
    }
    //cout<<"bin "<<i<<" eta: "<<xcent<<" mean: "<<mean<<endl;
    mean  = hMeansHistNegTOF->GetBinContent(i);
    xcent = hMeansHistNegTOF->GetBinCenter(i);
    if(fabs(mean)!=0){
      graphMeanTOFNeg->SetPoint(iTOFneg,xcent,mean);
      iTOFneg++;
    }
  }
  //-------------------------------------









  TCanvas *Canvas = GetCanvas(Form("CNsimavsBG%s",name.Data()),xpos,ypos,cansizeX,cansizeY,0,0,0.02,0.02,0.14,0.01);
  Canvas->cd();
  //Func: GetPad(name.Data(), xpos1, ypos1, xpos2, ypos2, topMar, botMar, leftMar, rightMar)
  TPad *padTop = GetPad(Form("padTop%s",name.Data()),0.0,0.771,1.0,1.0, 0.02,0.0,0.14,0.10);
  padTop->Draw();
  padTop->cd();
  padTop->SetTicks();
  padTop->SetLogx();
  padTop->SetLogz();
  padTop->SetGridy();
  hNSigTPCPos->SetTitle("");
  SetTitleTH1(hNSigTPCPos,"N#sigma_{TPC} +Ve",0.12,0.5,"#beta#gamma",0.08,xLblPos);
  SetAxisTH1(hNSigTPCPos,ySLow,ySHigh,xSLow,xSHigh,0.12,0.1,0.085,-0.01);
  hNSigTPCPos->Draw("COLZ");

  SetMarkerTH1(graphMeanTPCPos,"",25,0.5,2,2);
  graphMeanTPCPos->Draw("PSAME");
  SetMarkerTH1(graphMeanTPCNeg,"",24,0.5,4,4);
  graphMeanTPCNeg->Draw("PSAME");

  ///--------- Markings in Extra momentum axis --------
  Float_t xBGLow = 0.2/fMass;
  Float_t xBGHigh= 4.0/fMass;
  Float_t yLinePos= 6.1; //10*funcPass2->Eval(1.0/fMass);


  ///The Axis, Start and End ticks:
  drawMyline(xBGLow,yLinePos,xBGHigh,yLinePos, 1, 2, 1);
  drawMyline(xBGLow,yLinePos,xBGLow,yLinePos-0.5, 1, 2, 1);
  drawMyline(xBGHigh,yLinePos,xBGHigh,yLinePos-0.5, 1, 2, 1);

  //drawMyText(Float_t xPos, Float_t yPos, Float_t size, TString text);
  drawMyText(xBGLow*0.9, yLinePos+0.1, 0.09, "0.2");
  drawMyText(xBGHigh*0.95, yLinePos+0.1, 0.09, "4");
  ///Axis name:
  drawMyText(xBGLow*1.05, yLinePos-1.1, 0.09, "p (GeV/c)");

  ///Some Extra ticks:
  xBGHigh= 0.5/fMass;
  drawMyline(xBGHigh,yLinePos,xBGHigh,yLinePos-0.5, 1, 2, 1);
  drawMyText(xBGHigh*0.95, yLinePos+0.1, 0.09, "0.5");

  xBGHigh= 1.0/fMass;
  drawMyline(xBGHigh,yLinePos,xBGHigh,yLinePos-0.5, 1, 2, 1);
  drawMyText(xBGHigh*0.95, yLinePos+0.1, 0.09, "1");

  xBGHigh= 2.0/fMass;
  drawMyline(xBGHigh,yLinePos,xBGHigh,yLinePos-0.5, 1, 2, 1);
  drawMyText(xBGHigh*0.95, yLinePos+0.1, 0.09, "2");

  //xBGHigh= 4.0/fMass;
  //drawMyline(xBGHigh,yLinePos,xBGHigh,yLinePos-0.5, 1, 2, 1);
  //drawMyText(xBGHigh*0.95, yLinePos+0.1, 0.09, "4");
  //------------------------------------------------------


  legend = new TLegend(0.65,0.75,0.875,0.95);
  legend->SetBorderSize(0);
  legend->SetFillColor(0);
  legend->SetTextFont(42);
  legend->SetTextSize(0.10);
  legend->AddEntry(hNSigTPCPos,Form("%s,",passName.Data()),"");
  legend->AddEntry(hNSigTPCPos,Form("%s",name.Data()),"");
  legend->Draw();


  Canvas->cd();
  //Func:  GetPad(name.Data(), xpos1, ypos1, xpos2, ypos2, topMar, botMar, leftMar, rightMar)
  TPad *pad2nd = GetPad(Form("pad2nd%s",name.Data()),0.0,0.521,1.0,0.771, 0.02,0.00,0.14,0.10);
  pad2nd->Draw();
  pad2nd->cd();
  pad2nd->SetTicks();
  pad2nd->SetLogx();
  pad2nd->SetLogz();
  pad2nd->SetGridy();
  hNSigTPCNeg->SetTitle("");
  SetTitleTH1(hNSigTPCNeg,"N#sigma_{TPC} -Ve",0.12,0.5,"#beta#gamma",0.08,xLblPos);
  SetAxisTH1(hNSigTPCNeg,ySLow,ySHigh,xSLow,xSHigh,0.12,0.1,0.085,-0.01);
  hNSigTPCNeg->Draw("COLZ");
  SetMarkerTH1(graphMeanTPCNeg,"",24,0.5,4,4);
  graphMeanTPCNeg->Draw("PSAME");

  //---------------------------------------------------
  ///The Axis, Start and End ticks:
  xBGLow = 0.2/fMass;
  xBGHigh= 4.0/fMass;

  drawMyline(xBGLow,yLinePos,xBGHigh,yLinePos, 1, 2, 1);
  drawMyline(xBGLow,yLinePos,xBGLow,yLinePos-0.5, 1, 2, 1);
  drawMyline(xBGHigh,yLinePos,xBGHigh,yLinePos-0.5, 1, 2, 1);

  //drawMyText(Float_t xPos, Float_t yPos, Float_t size, TString text);
  drawMyText(xBGLow*0.9, yLinePos+0.1, 0.09, "0.2");
  drawMyText(xBGHigh*0.95, yLinePos+0.1, 0.09, "4");
  ///Axis name:
  drawMyText(xBGLow*1.05, yLinePos-1.1, 0.09, "p (GeV/c)");

  ///Some Extra ticks:
  xBGHigh= 0.5/fMass;
  drawMyline(xBGHigh,yLinePos,xBGHigh,yLinePos-0.5, 1, 2, 1);
  drawMyText(xBGHigh*0.95, yLinePos+0.1, 0.09, "0.5");

  xBGHigh= 1.0/fMass;
  drawMyline(xBGHigh,yLinePos,xBGHigh,yLinePos-0.5, 1, 2, 1);
  drawMyText(xBGHigh*0.95, yLinePos+0.1, 0.09, "1");

  xBGHigh= 2.0/fMass;
  drawMyline(xBGHigh,yLinePos,xBGHigh,yLinePos-0.5, 1, 2, 1);
  drawMyText(xBGHigh*0.95, yLinePos+0.1, 0.09, "2");
  //------------------------------------------------------




  Canvas->cd();
  TPad *pad3rd = GetPad(Form("pad3rd%s",name.Data()),0.0,0.276,1.0,0.521, 0.02,0.0,0.14,0.10);
  pad3rd->Draw();
  pad3rd->cd();
  pad3rd->SetTicks();
  pad3rd->SetLogx();
  pad3rd->SetLogz();
  pad3rd->SetGridy();
  hNSigTOFPos->SetTitle("");
  SetTitleTH1(hNSigTOFPos,"N#sigma_{TOF} +Ve",0.12,0.5,"#beta#gamma",0.06,xLblPos);
  SetAxisTH1(hNSigTOFPos,ySLow,ySHigh,xSLow,xSHigh,0.12,0.1,0.085,-0.01);
  hNSigTOFPos->Draw("COLZ");
  SetMarkerTH1(graphMeanTOFPos,"",25,0.5,2,2);
  graphMeanTOFPos->Draw("PSAME");
  SetMarkerTH1(graphMeanTOFNeg,"",24,0.5,4,4);
  graphMeanTOFNeg->Draw("PSAME");


  ///The Axis, Start and End ticks:
  xBGLow = 0.2/fMass;
  xBGHigh= 4.0/fMass;

  drawMyline(xBGLow,yLinePos,xBGHigh,yLinePos, 1, 2, 1);
  drawMyline(xBGLow,yLinePos,xBGLow,yLinePos-0.5, 1, 2, 1);
  drawMyline(xBGHigh,yLinePos,xBGHigh,yLinePos-0.5, 1, 2, 1);

  //drawMyText(Float_t xPos, Float_t yPos, Float_t size, TString text);
  drawMyText(xBGLow*0.9, yLinePos+0.1, 0.09, "0.2");
  drawMyText(xBGHigh*0.95, yLinePos+0.1, 0.09, "4");
  ///Axis name:
  drawMyText(xBGLow*1.05, yLinePos-1.1, 0.09, "p (GeV/c)");

  ///Some Extra ticks:
  xBGHigh= 0.5/fMass;
  drawMyline(xBGHigh,yLinePos,xBGHigh,yLinePos-0.5, 1, 2, 1);
  drawMyText(xBGHigh*0.95, yLinePos+0.1, 0.09, "0.5");

  xBGHigh= 1.0/fMass;
  drawMyline(xBGHigh,yLinePos,xBGHigh,yLinePos-0.5, 1, 2, 1);
  drawMyText(xBGHigh*0.95, yLinePos+0.1, 0.09, "1");

  xBGHigh= 2.0/fMass;
  drawMyline(xBGHigh,yLinePos,xBGHigh,yLinePos-0.5, 1, 2, 1);
  drawMyText(xBGHigh*0.95, yLinePos+0.1, 0.09, "2");
  //------------------------------------------------------



  Canvas->cd();
  TPad *pad4th = GetPad(Form("pad4th%s",name.Data()),0.0,0.0,1.0,0.276, 0.02,0.20,0.14,0.10);
  pad4th->Draw();
  pad4th->cd();
  pad4th->SetTicks();
  pad4th->SetLogx();
  pad4th->SetLogz();
  pad4th->SetGridy();
  hNSigTOFNeg->SetTitle("");
  SetTitleTH1(hNSigTOFNeg,"N#sigma_{TOF} -Ve",0.11,0.5,"#beta#gamma",0.10,xLblPos);
  SetAxisTH1(hNSigTOFNeg,ySLow,ySHigh,xSLow,xSHigh,0.11,0.1,0.085,-0.01);
  hNSigTOFNeg->Draw("COLZ");
  graphMeanTOFNeg->Draw("PSAME");

  ///--------- Markings in Extra momentum axis --------
  xBGLow = 0.2/fMass;
  xBGHigh= 4.0/fMass;
  ///The Axis, Start and End ticks:
  drawMyline(xBGLow,yLinePos,xBGHigh,yLinePos, 1, 2, 1);
  drawMyline(xBGLow,yLinePos,xBGLow,yLinePos-0.5, 1, 2, 1);
  drawMyline(xBGHigh,yLinePos,xBGHigh,yLinePos-0.5, 1, 2, 1);

  //drawMyText(Float_t xPos, Float_t yPos, Float_t size, TString text);
  drawMyText(xBGLow*0.9, yLinePos+0.1, 0.08, "0.2");
  drawMyText(xBGHigh*0.95, yLinePos+0.1, 0.08, "4");
  ///Axis name:
  drawMyText(xBGLow*1.05, yLinePos-1.1, 0.08, "p (GeV/c)");

  ///Some Extra ticks:
  xBGHigh= 0.5/fMass;
  drawMyline(xBGHigh,yLinePos,xBGHigh,yLinePos-0.5, 1, 2, 1);
  drawMyText(xBGHigh*0.95, yLinePos+0.1, 0.08, "0.5");

  xBGHigh= 1.0/fMass;
  drawMyline(xBGHigh,yLinePos,xBGHigh,yLinePos-0.5, 1, 2, 1);
  drawMyText(xBGHigh*0.95, yLinePos+0.1, 0.08, "1");

  xBGHigh= 2.0/fMass;
  drawMyline(xBGHigh,yLinePos,xBGHigh,yLinePos-0.5, 1, 2, 1);
  drawMyText(xBGHigh*0.95, yLinePos+0.1, 0.08, "2");
  //------------------------------------------------------



  Canvas->Update() ;

  if(Print)
    {
      std::string qaPath = CONFIG["output"]["skimTreeQA"]["QApath"].get<std::string>();
      TString qaPathT = TString::Format("%s", qaPath.c_str());
      if (!qaPathT.EndsWith("/")) qaPathT += "/";
      Canvas->SaveAs(Form("%sNsigmavsBetaGamma_%s%s.pdf", qaPathT.Data(), name.Data(), passName.Data()));
    }
}













