#include "../utils/headerfunction.h"
#include "./read_config.C"
// This macro was modified to automatically use the provided pathtoskimtree when entering the name (f.e. LHC23zzk)

///Global Parameter for Canvas
Int_t  xpos=30, ypos=30;
Int_t  cansizeX=550;
Int_t  cansizeY=700;
Float_t xLabelPos = 0.5;


// functions to do monkey work:
void plotPID2D(TH2 *hdEdx2Ddata=0x0, TH1 *hdEdxMeanData=0x0, TH1 *hdEdxMeanDataNeg=0x0, TString name="", Float_t fMass=1.0, int iMarker=24, int ci=1, TString passName="", TF1 *funcPass3=0x0, TF1* funcPass2=0x0, Double_t *scores=0x0, Float_t ySLow=0.1, Float_t ySHigh=1E4,Bool_t Print=kFALSE);
void plotPID2DMerged(TH2 *hdEdx2Ddata=0x0, TH1 *hdEdxMeanData=0x0, TH1 *hdEdxMeanDataNeg=0x0, TH1 *hdEdxMeanDataV0=0x0, TH1 *hdEdxMeanDataV0Neg=0x0, TString name="", Float_t fMass=1.0, int iMarker=24, int ci=1, TString passName="", TF1 *funcPass3=0x0, TF1* funcPass2=0x0, Double_t *scores=0x0, Float_t ySLow=0.1, Float_t ySHigh=1E4,Bool_t Print=kFALSE);


void plotNsigma(TH2 *hNSigTPCpos=0x0, TH2 *hNSigTPCneg=0x0, TH2 *hNSigTOFpos=0x0, TH2 *hNSigTOFneg=0x0, TString name="", Float_t fMass=1.0, TString passName="",Float_t ySLow=0.1, Float_t ySHigh=1E4,Bool_t Print=kFALSE);


//void drawMyline(Float_t xstrt=0,Float_t ystrt=0,Float_t xend=1,Float_t yend=0,Int_t iStyle=1,Int_t iWidth=1,Int_t icol=1);
//void drawMyText(Float_t xPos=1.0, Float_t yPos=1.0, Float_t size=0.1, TString text="");
//void drawMyTextNDC(Float_t xPos=1.0, Float_t yPos=1.0, Float_t size=0.1, TString text="");

static int aval = 1;
void RenormHistogram(TH1 *hInput=0x0, TF1 *funcBB=0x0);
void fillMyGraph(TGraphErrors *g1=0x0, TH1* hInput=0x0, double xLow=0.2, double xHigh=2E4, int &count=aval);
void fillMyGraphReNorm(TGraphErrors *g1=0x0, TH1* hInput=0x0, TF1* funcBB=0x0, double xLow=0.2, double xHigh=2E4, int &count=aval);
void fillMyGraphReNorm(TGraphErrors  *g1=0x0, TH1* hInput=0x0, TH1* hError=0x0,TF1* funcBB=0x0, double xLow=0.2, double xHigh=2E4, int &count=aval);
  
void fillMyGraph(TGraphErrors *g1=0x0, TH1* hInput=0x0, TH1* hError=0x0, int &count=aval);
void fillMyGraph(TGraphErrors *g1=0x0, TH1* hInput=0x0, TH1* hError=0x0, double xLow=0.2, double xHigh=2E4, int &count=aval);
Double_t GetScoreRatio(TH1* hInput=0x0);


//// main plotting macro starts here:


void fitNormGraphdEdxvsBGpid_modified(){

  readConfig();

  // Construct the dataset name and path from JSON config
  // std::string cfgYear = CONFIG["dataset"]["year"].get<std::string>();
  // std::string cfgPeriod = CONFIG["dataset"]["period"].get<std::string>();
  // std::string cfgPass = CONFIG["dataset"]["pass"].get<std::string>();
  std::string cfgDedxSelection = CONFIG["dataset"]["dEdxSelection"].get<std::string>();
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
  
  Bool_t LimEntryFit = kFALSE; // Read the following:
  // Warning Limit Entries of pi,k,p,e to ~2 Mil per species for the TGraphFit
  // as some trees (without downsampling / improper downsampling may have huge
  //(~50 Mil or more) tracks which may slowdown running this macro.

  
  Bool_t clusterCut =kTRUE; //do a cluster cut at 120
  // Int_t sectorA11cut= 1; //1-> doesn't include A11 sector, 2-> only A11 sector, any other number-> no cuts on A11 sector

  TLatex  *tex;
  TLegend *legend;
  Int_t runToSkip = -1; // 544116;  ///-1 = do not skip any run. <runnumber> =  Skip this particular run. 
  TString runFill="";   //Just to print in Canvas Legends. Example:
  //TString runFill = "537(734-855)"; //"LHC23za"
  
  TString fileName = TString::Format("%s", path2file.Data());

  cout << "Looking for file: " << fileName.Data() << endl;
  
  cout << "This is the filename " << fileName.Data() << endl;
  cout << "This is the sDataSet " << sDataSet.Data() << endl;
  

  // Open the file
  ifstream intxt;
  intxt.open(fileName.Data());  // Open the dynamically created file
  
  if (!intxt.is_open()) {
    cout << "Error: Could not open file: " << fileName.Data() << endl;
    return;
  }

    // Define the vectors
    std::vector<std::string> before_issues = {"LHC23zzf", "LHC23zzg", "LHC23zzh", "LHC23zzi"};
    std::vector<std::string> with_issues = {"LHC23zzk", "LHC23zzl", "LHC23zzm", "LHC23zzn", "LHC23zzo"};
    
    // Example dataset to check
    // TString sDataSet = "LHC23zzk";
    Int_t sectorA11cut = 0; // 1 -> doesn't include A11 sector, 2 -> only A11 sector, any other number -> no cuts on A11 sector
    
    // Check in which array the dataset exists
    if (std::find(before_issues.begin(), before_issues.end(), sDataSet.Data()) != before_issues.end()) {
        std::cout << sDataSet.Data() << " is from before the A11 issues. All sectors will be used." << std::endl;
    } else if (std::find(with_issues.begin(), with_issues.end(), sDataSet.Data()) != with_issues.end()) {
        std::cout << sDataSet.Data() << " is from a run with A11 issues. The sector is cut out." << std::endl;
        sectorA11cut = 1;
    } else {
        std::cout << sDataSet.Data() << " is from another run. All data will be used." << std::endl;
    }
    
    // Output the selected cut value
    std::cout << "Sector A11 cut value chosen: " << sectorA11cut << std::endl;

  // Read the file path from the file
  intxt >> path2file;  // Read the full path from the file
  cout << "Base path read from " << fileName.Data() << ": " << path2file.Data() << endl;

  // You can now use the path2file variable to load the dataset, etc.
  TFile *f1 = TFile::Open(Form("%s", fileName.Data()), "READ");
  if (!f1 || f1->IsZombie()) {
    cout << "Error opening file: " << fileName.Data() << endl;
    return;
  }
  
  

  //Default Parameters Loaded from CCDB (Thanks to Jeremy). In terminal window with Root session (disabled https and http)
  //o2-pidparam-tpc-response --mode pull --min-runnumber < 5xxxxx >

  Double_t OptParOld[6] = {0.280991, 3.23543, 0.0244042, 2.31595, 0.780374}; 
  Double_t OptParInit[6] = {0.280991, 3.23543, 0.0244042, 2.31595, 0.780374}; // default for LHC23 period.
 
  
  ///// Define the Bethe-Bloch Functions:  
  TF1 *funcBBvsBGDefault =  new TF1("funcBBvsBGDefault","[0]*([1]- log([2]+pow(x,-1.*[4])) - pow((x/sqrt(1+x*x)),[3]))*pow(1.0,2.3)*50./pow((x/sqrt(1+x*x)),[3])",2.e-3,4.E5);
  funcBBvsBGDefault->SetParameters(OptParOld); /// This is only used for Comparison.
  
  TF1 *funcBBvsBGReference = new TF1("funcBBvsBGReference","[0]*([1]- log([2]+pow(x,-1.*[4])) - pow((x/sqrt(1+x*x)),[3]))*pow(1.0,2.3)*50./pow((x/sqrt(1+x*x)),[3])",2.e-3,4.E5);
  funcBBvsBGReference->SetParameters(OptParInit); /// This is only used for Normalizing.
  


  //This function is used for Fit!!  
  TF1 *funcBBvsBGThisPass = new TF1("funcBBvsBGThisPass","[0]*([1]- log([2]+pow(x,-1.*[4])) - pow((x/sqrt(1+x*x)),[3]))*pow(1.0,2.3)*50./pow((x/sqrt(1+x*x)),[3])",0.005,2.E5); funcBBvsBGThisPass->SetParameters(OptParInit);

  funcBBvsBGThisPass->SetParLimits(2,OptParInit[2]*0.005,OptParInit[2]*100);  //some wide limits to stop from run-away parameters!
  funcBBvsBGThisPass->SetLineColor(4);
  
  if(!f1){
    cout<<" ::: ERROR ::: \n Could not find the file, Check name or Path! \n Exit!"<<endl; return;
  }
  
  if(f1->IsOpen()){
    cout<<" Opening File: "<<f1->GetName()<<endl;
  }
  else{
    cout<<" ::: WARNING ::: \n Could not open the file! check path or if file is corrupt? \n Exit!"<<endl; return;
  }

  
  TIter keyList(f1->GetListOfKeys());

  TKey    *key;
  TTree   *fChain;

  /// Now read the branches:

  UChar_t fPidIndex;
  Int_t   fRunNumber;
  Float_t fNSigTPC;
  Float_t fNSigTOF;
  Float_t fTPCSignal;
  Float_t fTPCdEdxNorm;
  Float_t fBetaGamma;
  Float_t fSigned1Pt;
  Float_t fTPCInnerMom;
  Float_t fInvDeDxExpTPC;
  Float_t fEta, fPhi;
  Float_t fNormNClustersTPC;   
  Float_t fFt0Occ;


  TBranch *b_fNSigTPC;   //!
  TBranch *b_fNSigTOF;   //!
  TBranch *b_fPidIndex;    //!
  TBranch *b_fBetaGamma;   //!
  TBranch *b_fRunNumber;   //!
  TBranch *b_fTPCSignal;   //!  
  TBranch *b_fTPCdEdxNorm;   //!  
  TBranch *b_fSigned1Pt;   //!
  TBranch *b_fTPCInnerParam; //!
  TBranch *b_fInvDeDxExpTPC; //!
  TBranch *b_fEta;         //!
  TBranch *b_fPhi;         //!
  TBranch *b_fNormNClustersTPC;         //!
  TBranch *b_fFt0Occ;   //!

  

  Double_t binEdge = 0.0;
  Double_t width = 0.01; //0.001
  const Int_t icNbinX = 140;  
  Double_t profBins[icNbinX+1] = {0,};
  profBins[0]={0.005};
  
  cout<<"bins: ";
  for(int i=0;i<icNbinX;i++){
    //width = width*exp(0.1215);
    width = width*exp(0.105);
    cout<<binEdge+width<<", ";
    profBins[i+1] = binEdge+width;
  }
  cout<<endl;
 
  //return;



  
  TH2F *hdEdxvsMomDataPion   =  new TH2F("hdEdxvsMomDataPionPos","dE/dx vs #beta#gamma #pi^{#pm}(Data)",icNbinX,profBins,4000,0,4000);  
  TH2F *hdEdxvsMomDataKaon   =  new TH2F("hdEdxvsMomDataKaonPos","dE/dx vs #beta#gamma  K^{#pm} (Data)",icNbinX,profBins,4000,0,4000);
  TH2F *hdEdxvsMomDataProt   =  new TH2F("hdEdxvsMomDataProtPos","dE/dx vs #beta#gamma p#bar{p} (Data)",icNbinX,profBins,5000,0,5000);
  TH2F *hdEdxvsMomDataPionNeg = new TH2F("hdEdxvsMomDataPionNeg","dE/dx vs #beta#gamma #pi^{#pm}(Data)",icNbinX,profBins,4000,0,4000);  
  TH2F *hdEdxvsMomDataKaonNeg = new TH2F("hdEdxvsMomDataKaonNeg","dE/dx vs #beta#gamma  K^{#pm} (Data)",icNbinX,profBins,4000,0,4000);
  TH2F *hdEdxvsMomDataProtNeg = new TH2F("hdEdxvsMomDataProtNeg","dE/dx vs #beta#gamma p#bar{p} (Data)",icNbinX,profBins,4000,0,4000);

  TH2F *hdEdxvsMomDataV0Elec   =  new TH2F("hdEdxvsMomDataV0ElecPos","dE/dx vs #beta#gamma  e^{#pm}  (Data)",icNbinX,profBins,4000,0,4000);
  TH2F *hdEdxvsMomDataV0Pion   =  new TH2F("hdEdxvsMomDataV0PionPos","dE/dx vs #beta#gamma #pi^{#pm} (Data)",icNbinX,profBins,4000,0,4000);
  TH2F *hdEdxvsMomDataV0Prot   =  new TH2F("hdEdxvsMomDataV0ProtPos","dE/dx vs #beta#gamma  p#bar{p} (Data)",icNbinX,profBins,4000,0,4000);
  TH2F *hdEdxvsMomDataV0Kaon   =  new TH2F("hdEdxvsMomDataV0KaonPos","dE/dx vs #beta#gamma  K^{#pm} (Data)",icNbinX,profBins,4000,0,4000);
  TH2F *hdEdxvsMomDataV0ElecNeg = new TH2F("hdEdxvsMomDataV0ElecNeg","dE/dx vs #beta#gamma  e^{#pm}  (Data)",icNbinX,profBins,4000,0,4000);
  TH2F *hdEdxvsMomDataV0PionNeg = new TH2F("hdEdxvsMomDataV0PionNeg","dE/dx vs #beta#gamma #pi^{#pm} (Data)",icNbinX,profBins,4000,0,4000);
  TH2F *hdEdxvsMomDataV0ProtNeg = new TH2F("hdEdxvsMomDataV0ProtNeg","dE/dx vs #beta#gamma  p#bar{p} (Data)",icNbinX,profBins,5000,0,5000);      
  TH2F *hdEdxvsMomDataV0KaonNeg = new TH2F("hdEdxvsMomDataV0KaonNeg","dE/dx vs #beta#gamma  K^{#pm} (Data)",icNbinX,profBins,4000,0,4000);      

  //Selected with TPC,TOF 
  TH2F *hNormdEdxvsMomDataPion   =  new TH2F("hNormdEdxvsMomDataPionTPCTOFPos","dE/dx vs #beta#gamma #pi^{+}(Data)",icNbinX,profBins,200,0.5,1.5);  
  TH2F *hNormdEdxvsMomDataKaon   =  new TH2F("hNormdEdxvsMomDataKaonTPCTOFPos","dE/dx vs #beta#gamma  K^{+} (Data)",icNbinX,profBins,200,0.5,1.5);
  TH2F *hNormdEdxvsMomDataProt   =  new TH2F("hNormdEdxvsMomDataProtTPCTOFPos","dE/dx vs #beta#gamma   prot (Data)",icNbinX,profBins,200,0.5,1.5);
  TH2F *hNormdEdxvsMomDataPionNeg = new TH2F("hNormdEdxvsMomDataPionTPCTOFNeg","dE/dx vs #beta#gamma #pi^{-}(Data)",icNbinX,profBins,200,0.5,1.5);  
  TH2F *hNormdEdxvsMomDataKaonNeg = new TH2F("hNormdEdxvsMomDataKaonTPCTOFNeg","dE/dx vs #beta#gamma  K^{-} (Data)",icNbinX,profBins,200,0.5,1.5);
  TH2F *hNormdEdxvsMomDataProtNeg = new TH2F("hNormdEdxvsMomDataProtTPCTOFNeg","dE/dx vs #beta#gamma #bar{p}(Data)",icNbinX,profBins,200,0.5,1.5);
  
  //Selected with V0 
  TH2F *hNormdEdxvsMomDataV0Elec   =  new TH2F("hNormdEdxvsMomDataV0ElecPos","dE/dx vs #beta#gamma  e^{+}  (Data)",icNbinX,profBins,200,0.5,1.5);
  TH2F *hNormdEdxvsMomDataV0Pion   =  new TH2F("hNormdEdxvsMomDataV0PionPos","dE/dx vs #beta#gamma #pi^{+} (Data)",icNbinX,profBins,200,0.5,1.5);
  TH2F *hNormdEdxvsMomDataV0Prot   =  new TH2F("hNormdEdxvsMomDataV0ProtPos","dE/dx vs #beta#gamma   prot  (Data)",icNbinX,profBins,200,0.5,1.5);
  TH2F *hNormdEdxvsMomDataV0Kaon   =  new TH2F("hNormdEdxvsMomDataV0KaonPos","dE/dx vs #beta#gamma  K^{+}  (Data)",icNbinX,profBins,200,0.5,1.5);
  TH2F *hNormdEdxvsMomDataV0ElecNeg = new TH2F("hNormdEdxvsMomDataV0ElecNeg","dE/dx vs #beta#gamma  e^{-}  (Data)",icNbinX,profBins,200,0.5,1.5);
  TH2F *hNormdEdxvsMomDataV0PionNeg = new TH2F("hNormdEdxvsMomDataV0PionNeg","dE/dx vs #beta#gamma #pi^{-} (Data)",icNbinX,profBins,200,0.5,1.5);
  TH2F *hNormdEdxvsMomDataV0ProtNeg = new TH2F("hNormdEdxvsMomDataV0ProtNeg","dE/dx vs #beta#gamma #bar{p} (Data)",icNbinX,profBins,200,0.5,1.5);
  TH2F *hNormdEdxvsMomDataV0KaonNeg = new TH2F("hNormdEdxvsMomDataV0KaonNeg","dE/dx vs #beta#gamma  K^{-}  (Data)",icNbinX,profBins,200,0.5,1.5);


  //// Now Merging All Pions:
  TH2F *hdEdxDataPionPosMerged = new TH2F("hdEdxDataPionPosMerged","dE/dx vs #beta#gamma #pi^{+}(Data)",icNbinX,profBins,4000,0,4000); 
  TH2F *hdEdxDataPionNegMerged = new TH2F("hdEdxDataPionNegMerged","dE/dx vs #beta#gamma #pi^{-}(Data)",icNbinX,profBins,4000,0,4000); 

  TH2F *hdEdxDataProtPosMerged = new TH2F("hdEdxDataProtPosMerged","dE/dx vs #beta#gamma prot (Data)",icNbinX,profBins,4000,0,4000); 
  TH2F *hdEdxDataProtNegMerged = new TH2F("hdEdxDataProtNegMerged","dE/dx vs #beta#gamma pbar (Data)",icNbinX,profBins,4000,0,4000); 

  
  /// Nsigma Plots --> // Added a separate plotting macro for this: plotMacroSkimQA2.C




  
  TProfile *hdEdxVsBetaGammaTheo = new TProfile("hdEdxVsBetaGammaTheo","",icNbinX,profBins);
  Int_t ch = 0;
  UInt_t ipid;
  Int_t nTotTrk = 0;
  Int_t nV0Trk = 0;
  Int_t ntrkPion=0,ntrkKaon=0,ntrkProt=0,ntrkDeut=0;
  Int_t ntrkPionV0=0,ntrkElecV0=0,ntrkProtV0=0,ntrkKaonV0=0;
  
  Int_t oldRun = 0, thisRun;
  Double_t dBBtheo=0;

  
  cout<<"\n Reading Period: "<<sDataSet.Data();
  // cout<<"\n Using dEdxSelection value: "<<dEdxSelection;

  // checking weather to use the TPCSignal or TPCdEdxNorm
  // The values are replaced before filling the histograms
  if (cfgDedxSelection == "TPCdEdxNorm"){
    std::cout <<"\nFor the BB fitting the dEdx values from the TPCdEdxNorm branch are used." << std::endl;
  }
  else if (cfgDedxSelection == "TPCSignal")
  {
    std::cout <<"\nFor the BB fitting the dEdx values from the TPCSignal branch are used." << std::endl;
  }
  

  while ((key = (TKey*)keyList())) {
    
    TClass *cl = gROOT->GetClass(key->GetClassName());
    if (!cl->InheritsFrom("TDirectoryFile")) continue;
    TString dirName = (TString ) key->GetName();
  
    TTree *treeTPC=NULL;
    TTree *treeV0=NULL;
    
    TDirectory *dir = (TDirectory*)f1->Get(Form("%s:/%s",f1->GetName(),dirName.Data()));

    
    dir->GetObject(cgfTPCTOFtreename.c_str(),treeTPC);
   
    if (treeTPC){
      fChain = treeTPC;
      fChain->SetMakeClass(1);
      fChain->SetBranchAddress("fRunNumber", &fRunNumber, &b_fRunNumber);
      fChain->SetBranchAddress("fNSigTPC", &fNSigTPC, &b_fNSigTPC);
      fChain->SetBranchAddress("fNSigTOF", &fNSigTOF, &b_fNSigTOF);
      fChain->SetBranchAddress("fPidIndex", &fPidIndex, &b_fPidIndex);
      fChain->SetBranchAddress("fBetaGamma", &fBetaGamma, &b_fBetaGamma);
      fChain->SetBranchAddress("fTPCSignal", &fTPCSignal, &b_fTPCSignal);
      fChain->SetBranchAddress("fTPCdEdxNorm", &fTPCdEdxNorm, &b_fTPCdEdxNorm);
      fChain->SetBranchAddress("fSigned1Pt", &fSigned1Pt, &b_fSigned1Pt);
      fChain->SetBranchAddress("fTPCInnerParam", &fTPCInnerMom, &b_fTPCInnerParam);
      fChain->SetBranchAddress("fInvDeDxExpTPC", &fInvDeDxExpTPC, &b_fInvDeDxExpTPC);
      fChain->SetBranchAddress("fEta", &fEta, &b_fEta);
      fChain->SetBranchAddress("fPhi", &fPhi, &b_fPhi);
      fChain->SetBranchAddress("fNormNClustersTPC", &fNormNClustersTPC, &b_fNormNClustersTPC);
      fChain->SetBranchAddress("fFt0Occ", &fFt0Occ, &b_fFt0Occ);



      
      Long64_t nTrk = fChain->GetEntries();
      cout<<"\n tpc-tof Dir name: "<<dirName.Data()<<", Entries: "<<nTrk<<endl;
      cout<<" Reading Runs: \n ";
      
      for (Int_t i = 0; i < nTrk; i++) {
	    fChain->GetEntry(i);
	

	if(runToSkip>0)
	  if(fRunNumber==runToSkip) continue;

	if(LimEntryFit && fabs(ipid)==2 && ntrkPion>4E6) continue;
	if(LimEntryFit && fabs(ipid)==3 && ntrkKaon>4E6) continue;
	if(LimEntryFit && fabs(ipid)==4 && ntrkProt>4E6) continue;	
	if(LimEntryFit &&  ntrkPion>4E6 && ntrkKaon>4E6 && ntrkProt>4E6) break; 



	
	ipid = unsigned(fPidIndex);
	if(fSigned1Pt>0) ch=1;
	else ch=-1;

	if(oldRun!=fRunNumber){
	  oldRun = fRunNumber;
	  cout<<oldRun<<", ";
	}

//---------------------------------------------------------------------
  //condition to remove A11 sector for periods LHC23zzk,zzl,zzm,zzn,zzo

  if(sectorA11cut == 1){
    if (fEta>0 && fPhi > (2* (M_PI/18) * 10.5) && fPhi < (2* (M_PI/18) * 12.5)) continue;
  }
  
  else if(sectorA11cut == 2){
    if (!(fEta>0 && fPhi > (2* (M_PI/18) * 11) && fPhi < (2* (M_PI/18) * 12))) continue;
  } 
  else {}

  if (clusterCut == kTRUE)
  if ((152/(fNormNClustersTPC*fNormNClustersTPC))<120)continue;

	dBBtheo = funcBBvsBGReference->Eval(fBetaGamma);

  // In case of usage of the TPCdEdxNorm, replace fTPCSignal with fTPCdEdxNorm
  if (cfgDedxSelection == "TPCdEdxNorm"){
    fTPCSignal = fTPCdEdxNorm;
  }
	if(ch>0){
	  if(fabs(ipid)==2 && fTPCInnerMom < 1.8){
	    hdEdxvsMomDataPion->Fill(fBetaGamma,fTPCSignal);
	    hdEdxDataPionPosMerged->Fill(fBetaGamma,fTPCSignal);
	    hNormdEdxvsMomDataPion->Fill(fBetaGamma,fTPCSignal/dBBtheo);
	    ntrkPion++;
	  }
	  else if(fabs(ipid)==3){
	    hdEdxvsMomDataKaon->Fill(fBetaGamma,fTPCSignal);
	    hNormdEdxvsMomDataKaon->Fill(fBetaGamma,fTPCSignal/dBBtheo);
	    ntrkKaon++;
	  }
	  else if(fabs(ipid)==4 && fTPCInnerMom < 0.8){
	    hdEdxvsMomDataProt->Fill(fBetaGamma,fTPCSignal);
	    hdEdxDataProtPosMerged->Fill(fBetaGamma,fTPCSignal);
	    hNormdEdxvsMomDataProt->Fill(fBetaGamma,fTPCSignal/dBBtheo);
	    ntrkProt++;
	  }
	}
	else{
	  if(fabs(ipid)==2 && fTPCInnerMom < 1.8){
	    hdEdxvsMomDataPionNeg->Fill(fBetaGamma,fTPCSignal);
	    hdEdxDataPionNegMerged->Fill(fBetaGamma,fTPCSignal);
	    hNormdEdxvsMomDataPionNeg->Fill(fBetaGamma,fTPCSignal/dBBtheo);
	    ntrkPion++;
	  }
	  else if(fabs(ipid)==3){
	    hdEdxvsMomDataKaonNeg->Fill(fBetaGamma,fTPCSignal);
	    hNormdEdxvsMomDataKaonNeg->Fill(fBetaGamma,fTPCSignal/dBBtheo);
	    ntrkKaon++;
	  }
	  else if(fabs(ipid)==4 && fTPCInnerMom < 0.8){
	    hdEdxvsMomDataProtNeg->Fill(fBetaGamma,fTPCSignal);
	    hdEdxDataProtNegMerged->Fill(fBetaGamma,fTPCSignal);
	    hNormdEdxvsMomDataProtNeg->Fill(fBetaGamma,fTPCSignal/dBBtheo);
	    ntrkProt++;
	  }
	}	
	
	nTotTrk++;
      }//track loop      
    }//directory loop



    ///Now Read the V0 tree:
    
    dir->GetObject(cgfV0treename.c_str(),treeV0);
    
    if(treeV0){
      //cout<<" Found the V0 Tree "<<endl;
      fChain = treeV0;
      fChain->SetMakeClass(1);
      fChain->SetBranchAddress("fRunNumber", &fRunNumber, &b_fRunNumber);
      fChain->SetBranchAddress("fNSigTPC", &fNSigTPC, &b_fNSigTPC);
      fChain->SetBranchAddress("fNSigTOF", &fNSigTOF, &b_fNSigTOF);
      fChain->SetBranchAddress("fPidIndex", &fPidIndex, &b_fPidIndex);
      fChain->SetBranchAddress("fTPCSignal", &fTPCSignal, &b_fTPCSignal);
      fChain->SetBranchAddress("fTPCdEdxNorm", &fTPCdEdxNorm, &b_fTPCdEdxNorm);
      fChain->SetBranchAddress("fBetaGamma", &fBetaGamma, &b_fBetaGamma);
      fChain->SetBranchAddress("fSigned1Pt", &fSigned1Pt, &b_fSigned1Pt);
      fChain->SetBranchAddress("fTPCInnerParam", &fTPCInnerMom, &b_fTPCInnerParam);
      fChain->SetBranchAddress("fInvDeDxExpTPC", &fInvDeDxExpTPC, &b_fInvDeDxExpTPC);
      fChain->SetBranchAddress("fEta", &fEta, &b_fEta);
      fChain->SetBranchAddress("fPhi", &fPhi, &b_fPhi);
      fChain->SetBranchAddress("fNormNClustersTPC", &fNormNClustersTPC, &b_fNormNClustersTPC);
      fChain->SetBranchAddress("fFt0Occ", &fFt0Occ, &b_fFt0Occ);



      
      Long64_t nTrk = fChain->GetEntries();
      //cout<<"name: "<<dirName.Data()<<" Entries: "<<nTrk<<endl;
      cout<<"\n V0sel Dir name: "<<dirName.Data()<<", Entries: "<<nTrk<<endl;

	   
      for (Int_t i = 0; i < nTrk; i++) {	
	fChain->GetEntry(i);

	if(runToSkip>0)
	  if(fRunNumber==runToSkip) continue;

	if(LimEntryFit &&  fabs(ipid)==0 &&  ntrkElecV0>4E6) continue;
	if(LimEntryFit &&  fabs(ipid)==2 &&  ntrkPionV0>4E6) continue;
	if(LimEntryFit &&  fabs(ipid)==4 &&  ntrkProtV0>4E6) continue;
	if(LimEntryFit && ntrkPionV0>4E6 && ntrkElecV0>4E6 && ntrkProtV0>4E6) break;
	
	
	if(fSigned1Pt>0) ch=1;
	else ch=-1;
	
	ipid = unsigned(fPidIndex);
  

 //condition to remove A11 sector for periods LHC23zzk,zzl,zzm,zzn,zzo

  if(sectorA11cut == 1){
    if (fEta>0 && fPhi > (2* (M_PI/18) * 10.5) && fPhi < (2* (M_PI/18) * 12.5)) continue;
  }
  
  else if(sectorA11cut == 2){
    if (!(fEta>0 && fPhi > (2* (M_PI/18) * 11) && fPhi < (2* (M_PI/18) * 12))) continue;
  } 
  else {}

  if (clusterCut == kTRUE)
  if ((152/(fNormNClustersTPC*fNormNClustersTPC))<120) continue;


	dBBtheo = funcBBvsBGReference->Eval(fBetaGamma);
	// In case of usage of the TPCdEdxNorm, replace fTPCSignal with fTPCdEdxNorm
  if (cfgDedxSelection == "TPCdEdxNorm"){
    fTPCSignal = fTPCdEdxNorm;
  }
	if(ch>0){
	  if(fabs(ipid)==0 && fabs(fNSigTPC)<3.0){
	    hdEdxvsMomDataV0Elec->Fill(fBetaGamma,fTPCSignal);
	    hNormdEdxvsMomDataV0Elec->Fill(fBetaGamma,fTPCSignal/dBBtheo);
	    ntrkElecV0++;
	  }
	  else if(fabs(ipid)==2 && fabs(fNSigTPC)<5.0){
	    hdEdxvsMomDataV0Pion->Fill(fBetaGamma,fTPCSignal);
	    hdEdxDataPionPosMerged->Fill(fBetaGamma,fTPCSignal);
	    hNormdEdxvsMomDataV0Pion->Fill(fBetaGamma,fTPCSignal/dBBtheo);
	    ntrkPionV0++;
	  }
    else if(fabs(ipid)==3 && fabs(fNSigTPC)<5.0){
      hdEdxvsMomDataV0Kaon->Fill(fBetaGamma,fTPCSignal);
      hNormdEdxvsMomDataV0Kaon->Fill(fBetaGamma,fTPCSignal/dBBtheo);
      ntrkKaonV0++;
    }
	  else if(fabs(ipid)==4 && fabs(fNSigTPC)<5.0){
	    hdEdxvsMomDataV0Prot->Fill(fBetaGamma,fTPCSignal);
	    hdEdxDataProtPosMerged->Fill(fBetaGamma,fTPCSignal);
	    hNormdEdxvsMomDataV0Prot->Fill(fBetaGamma,fTPCSignal/dBBtheo);
	    ntrkProtV0++;
	  }
	}
	else{
	  if(fabs(ipid)==0 && fabs(fNSigTPC)<3.0){
	    hdEdxvsMomDataV0ElecNeg->Fill(fBetaGamma,fTPCSignal);
	    hdEdxDataPionNegMerged->Fill(fBetaGamma,fTPCSignal);
	    hNormdEdxvsMomDataV0ElecNeg->Fill(fBetaGamma,fTPCSignal/dBBtheo);
	    ntrkElecV0++;
	  }
	  else if(fabs(ipid)==2 && fabs(fNSigTPC)<5.0){
	    hdEdxvsMomDataV0PionNeg->Fill(fBetaGamma,fTPCSignal);
	    hNormdEdxvsMomDataV0PionNeg->Fill(fBetaGamma,fTPCSignal/dBBtheo);
	    ntrkPionV0++;
	  }
    else if(fabs(ipid)==3 && fabs(fNSigTPC)<5.0){
      hdEdxvsMomDataV0KaonNeg->Fill(fBetaGamma,fTPCSignal);
      hNormdEdxvsMomDataV0KaonNeg->Fill(fBetaGamma,fTPCSignal/dBBtheo);
      ntrkKaonV0++;
    }
	  else if(fabs(ipid)==4 && fabs(fNSigTPC)<5.0){
	    hdEdxvsMomDataV0ProtNeg->Fill(fBetaGamma,fTPCSignal);
	    hdEdxDataProtNegMerged->Fill(fBetaGamma,fTPCSignal);
	    hNormdEdxvsMomDataV0ProtNeg->Fill(fBetaGamma,fTPCSignal/dBBtheo);
	    ntrkProtV0++;
	  }
	}
	//hdEdxVsBetaGammaTheo->Fill(fBetaGamma,1./fInvDeDxExpTPC);
	nV0Trk++;
  }
 
    }
    
    
  }

    cout<<endl;


  //return;
  

  ///IN the following, we do the slice fitting:
  
  ///TPC-TOF selection, +Ve tracks:  
  hNormdEdxvsMomDataPion->FitSlicesY();
  TH1D *hMeandEdxvsMomDataPion  = (TH1D*) gDirectory->Get("hNormdEdxvsMomDataPionTPCTOFPos_1");
  TH1D *hSigmadEdxvsMomDataPion = (TH1D*) gDirectory->Get("hNormdEdxvsMomDataPionTPCTOFPos_2");
  hNormdEdxvsMomDataKaon->FitSlicesY();
  TH1D *hMeandEdxvsMomDataKaon  = (TH1D*) gDirectory->Get("hNormdEdxvsMomDataKaonTPCTOFPos_1");
  TH1D *hSigmadEdxvsMomDataKaon = (TH1D*) gDirectory->Get("hNormdEdxvsMomDataKaonTPCTOFPos_2");
  hNormdEdxvsMomDataProt->FitSlicesY();
  TH1D *hMeandEdxvsMomDataProt  = (TH1D*) gDirectory->Get("hNormdEdxvsMomDataProtTPCTOFPos_1");
  TH1D *hSigmadEdxvsMomDataProt = (TH1D*) gDirectory->Get("hNormdEdxvsMomDataProtTPCTOFPos_2");
  //TPC-TOF selection, -Ve tracks:  
  hNormdEdxvsMomDataPionNeg->FitSlicesY();
  TH1D *hMeandEdxvsMomDataPionNeg  = (TH1D*) gDirectory->Get("hNormdEdxvsMomDataPionTPCTOFNeg_1");
  TH1D *hSigmadEdxvsMomDataPionNeg = (TH1D*) gDirectory->Get("hNormdEdxvsMomDataPionTPCTOFNeg_2");
  hNormdEdxvsMomDataKaonNeg->FitSlicesY();
  TH1D *hMeandEdxvsMomDataKaonNeg  = (TH1D*) gDirectory->Get("hNormdEdxvsMomDataKaonTPCTOFNeg_1");
  TH1D *hSigmadEdxvsMomDataKaonNeg = (TH1D*) gDirectory->Get("hNormdEdxvsMomDataKaonTPCTOFNeg_2");
  hNormdEdxvsMomDataProtNeg->FitSlicesY();
  TH1D *hMeandEdxvsMomDataProtNeg  = (TH1D*) gDirectory->Get("hNormdEdxvsMomDataProtTPCTOFNeg_1");
  TH1D *hSigmadEdxvsMomDataProtNeg = (TH1D*) gDirectory->Get("hNormdEdxvsMomDataProtTPCTOFNeg_2");

  ///V0-Selections, +Ve tracks:  
  hNormdEdxvsMomDataV0Elec->FitSlicesY();
  TH1D *hMeandEdxvsMomDataV0Elec  = (TH1D*) gDirectory->Get("hNormdEdxvsMomDataV0ElecPos_1");
  TH1D *hSigmadEdxvsMomDataV0Elec = (TH1D*) gDirectory->Get("hNormdEdxvsMomDataV0ElecPos_2");
  hNormdEdxvsMomDataV0Pion->FitSlicesY();
  TH1D *hMeandEdxvsMomDataV0Pion  = (TH1D*) gDirectory->Get("hNormdEdxvsMomDataV0PionPos_1");
  TH1D *hSigmadEdxvsMomDataV0Pion = (TH1D*) gDirectory->Get("hNormdEdxvsMomDataV0PionPos_2");
  hNormdEdxvsMomDataV0Prot->FitSlicesY();
  TH1D *hMeandEdxvsMomDataV0Prot  = (TH1D*) gDirectory->Get("hNormdEdxvsMomDataV0ProtPos_1");
  TH1D *hSigmadEdxvsMomDataV0Prot = (TH1D*) gDirectory->Get("hNormdEdxvsMomDataV0ProtPos_2");
  hNormdEdxvsMomDataV0Kaon->FitSlicesY();
  TH1D *hMeandEdxvsMomDataV0Kaon  = (TH1D*) gDirectory->Get("hNormdEdxvsMomDataV0KaonPos_1");
  TH1D *hSigmadEdxvsMomDataV0Kaon = (TH1D*) gDirectory->Get("hNormdEdxvsMomDataV0KaonPos_2");
  //V0-Selections, -Ve tracks:
  hNormdEdxvsMomDataV0ElecNeg->FitSlicesY();
  TH1D *hMeandEdxvsMomDataV0ElecNeg  = (TH1D*) gDirectory->Get("hNormdEdxvsMomDataV0ElecNeg_1");
  TH1D *hSigmadEdxvsMomDataV0ElecNeg = (TH1D*) gDirectory->Get("hNormdEdxvsMomDataV0ElecNeg_2");
  hNormdEdxvsMomDataV0PionNeg->FitSlicesY();
  TH1D *hMeandEdxvsMomDataV0PionNeg  = (TH1D*) gDirectory->Get("hNormdEdxvsMomDataV0PionNeg_1");
  TH1D *hSigmadEdxvsMomDataV0PionNeg = (TH1D*) gDirectory->Get("hNormdEdxvsMomDataV0PionNeg_2");
  hNormdEdxvsMomDataV0ProtNeg->FitSlicesY();
  TH1D *hMeandEdxvsMomDataV0ProtNeg  = (TH1D*) gDirectory->Get("hNormdEdxvsMomDataV0ProtNeg_1");
  TH1D *hSigmadEdxvsMomDataV0ProtNeg = (TH1D*) gDirectory->Get("hNormdEdxvsMomDataV0ProtNeg_2");
  hNormdEdxvsMomDataV0KaonNeg->FitSlicesY();
  TH1D *hMeandEdxvsMomDataV0KaonNeg  = (TH1D*) gDirectory->Get("hNormdEdxvsMomDataV0KaonNeg_1");
  TH1D *hSigmadEdxvsMomDataV0KaonNeg = (TH1D*) gDirectory->Get("hNormdEdxvsMomDataV0KaonNeg_2");



  /// Set Ranges for the TGraphFit, to remove outlier bands.
  /// Now with the V0 and tpc-tof Pin Cuts in Place, these maynot be needed.
  /*
  hNormdEdxvsMomDataPion->GetXaxis()->SetRangeUser(0.025,38);
  hNormdEdxvsMomDataV0Pion->GetXaxis()->SetRangeUser(0.025,38);
  hNormdEdxvsMomDataPionNeg->GetXaxis()->SetRangeUser(0.025,38);
  hNormdEdxvsMomDataV0PionNeg->GetXaxis()->SetRangeUser(0.025,38);
  hNormdEdxvsMomDataKaon->GetXaxis()->SetRangeUser(0.025,2.8);
  hNormdEdxvsMomDataKaonNeg->GetXaxis()->SetRangeUser(0.025,2.8);
  hNormdEdxvsMomDataProt->GetXaxis()->SetRangeUser(0.005,7.1);
  hNormdEdxvsMomDataProtNeg->GetXaxis()->SetRangeUser(0.005,7.1);
  hNormdEdxvsMomDataV0Elec->GetXaxis()->SetRangeUser(0.025,3800);
  hNormdEdxvsMomDataV0ElecNeg->GetXaxis()->SetRangeUser(0.025,3800);
  */
    
  ///Write histograms into root file for Jens.
  /*
  TFile *fout = new TFile(Form("file2DNormdEdxvsBG_%s_NormMIP.root",sDataSet.Data()),"RECREATE");
  fout->cd();
  funcBBvsBGReference->Write();
  hNormdEdxvsMomDataPion->Write();
  hNormdEdxvsMomDataKaon->Write();
  hNormdEdxvsMomDataProt->Write();
  hNormdEdxvsMomDataPionNeg->Write();
  hNormdEdxvsMomDataKaonNeg->Write();
  hNormdEdxvsMomDataProtNeg->Write();
  
  hNormdEdxvsMomDataV0Elec->Write();
  hNormdEdxvsMomDataV0Pion->Write();
  hNormdEdxvsMomDataV0Prot->Write();
  hNormdEdxvsMomDataV0ElecNeg->Write();
  hNormdEdxvsMomDataV0PionNeg->Write();
  hNormdEdxvsMomDataV0ProtNeg->Write();
  */



  

  //// Now the Final work :
  TGraphErrors *graphdEdxMeanVsBG = new TGraphErrors();    /// All particles in one TGraph

  int ig = 0; // index for number of points in the graph. it starts with 0.

  ///Renormalize to get actual dEdx and then fill the graph:
  //V0 Particles:
  fillMyGraphReNorm(graphdEdxMeanVsBG, hMeandEdxvsMomDataV0Elec,   funcBBvsBGReference, 4E2, 2E4, ig);
  fillMyGraphReNorm(graphdEdxMeanVsBG, hMeandEdxvsMomDataV0ElecNeg,funcBBvsBGReference, 4E2, 2E4, ig);  
  fillMyGraphReNorm(graphdEdxMeanVsBG, hMeandEdxvsMomDataV0Pion,   funcBBvsBGReference, 1.4, 15.0,ig);
  fillMyGraphReNorm(graphdEdxMeanVsBG, hMeandEdxvsMomDataV0PionNeg,funcBBvsBGReference, 1.4, 15.0,ig);  
  fillMyGraphReNorm(graphdEdxMeanVsBG, hMeandEdxvsMomDataV0Prot,   funcBBvsBGReference, 0.10, 2.0, ig);  
  fillMyGraphReNorm(graphdEdxMeanVsBG, hMeandEdxvsMomDataV0ProtNeg,funcBBvsBGReference, 0.10, 2.0, ig);
  fillMyGraphReNorm(graphdEdxMeanVsBG, hMeandEdxvsMomDataV0Kaon,   funcBBvsBGReference, 0.40, 2.0, ig);
  fillMyGraphReNorm(graphdEdxMeanVsBG, hMeandEdxvsMomDataV0KaonNeg,funcBBvsBGReference, 0.40, 2.0, ig);
  ///TPC-TOF Particles:
  fillMyGraphReNorm(graphdEdxMeanVsBG, hMeandEdxvsMomDataKaon,   funcBBvsBGReference, 0.40, 2.00, ig);   
  fillMyGraphReNorm(graphdEdxMeanVsBG, hMeandEdxvsMomDataKaonNeg,funcBBvsBGReference, 0.40, 2.00, ig);   
  fillMyGraphReNorm(graphdEdxMeanVsBG, hMeandEdxvsMomDataPion,   funcBBvsBGReference, 1.4, 15.0,  ig);
  fillMyGraphReNorm(graphdEdxMeanVsBG, hMeandEdxvsMomDataPionNeg,funcBBvsBGReference, 1.4, 15.0, ig);  
  fillMyGraphReNorm(graphdEdxMeanVsBG, hMeandEdxvsMomDataProt,   funcBBvsBGReference, 0.10, 2.0, ig);  
  fillMyGraphReNorm(graphdEdxMeanVsBG, hMeandEdxvsMomDataProtNeg,funcBBvsBGReference, 0.10, 2.0, ig);
  



  

  ////---- Following graphs Are only for fancy figure (i.e., separate Graph for each species!). ------
  ///V0 Particles:
  TGraphErrors *graphdEdxMeanVsBGV0PionPos = new TGraphErrors();
  TGraphErrors *graphdEdxMeanVsBGV0ElecPos = new TGraphErrors();
  TGraphErrors *graphdEdxMeanVsBGV0ProtPos = new TGraphErrors();
  TGraphErrors *graphdEdxMeanVsBGV0PionNeg = new TGraphErrors();
  TGraphErrors *graphdEdxMeanVsBGV0ElecNeg = new TGraphErrors();
  TGraphErrors *graphdEdxMeanVsBGV0ProtNeg = new TGraphErrors();
  TGraphErrors *graphdEdxMeanVsBGV0KaonPos = new TGraphErrors();
  TGraphErrors *graphdEdxMeanVsBGV0KaonNeg = new TGraphErrors();

  Int_t iV0elp=0,iV0eln=0,iV0pip=0,iV0pin=0,iV0prp=0,iV0prn=0; 
  fillMyGraphReNorm(graphdEdxMeanVsBGV0ElecPos, hMeandEdxvsMomDataV0Elec,   hSigmadEdxvsMomDataV0Elec,   funcBBvsBGReference, 4E2, 10E4, iV0elp);
  fillMyGraphReNorm(graphdEdxMeanVsBGV0ElecNeg, hMeandEdxvsMomDataV0ElecNeg,hSigmadEdxvsMomDataV0ElecNeg,funcBBvsBGReference, 4E2, 10E4, iV0eln);
  fillMyGraphReNorm(graphdEdxMeanVsBGV0PionPos, hMeandEdxvsMomDataV0Pion,   hSigmadEdxvsMomDataV0Pion,   funcBBvsBGReference, 1.4, 15.0, iV0pip);
  fillMyGraphReNorm(graphdEdxMeanVsBGV0PionNeg, hMeandEdxvsMomDataV0PionNeg,hSigmadEdxvsMomDataV0PionNeg,funcBBvsBGReference, 1.4, 15.0, iV0pin);
  fillMyGraphReNorm(graphdEdxMeanVsBGV0ProtPos, hMeandEdxvsMomDataV0Prot,   hSigmadEdxvsMomDataV0Prot,   funcBBvsBGReference, 0.10, 2.0, iV0prp);
  fillMyGraphReNorm(graphdEdxMeanVsBGV0ProtNeg, hMeandEdxvsMomDataV0ProtNeg,hSigmadEdxvsMomDataV0ProtNeg,funcBBvsBGReference, 0.10, 2.0, iV0prn);
  Int_t iV0Kap=0,iV0Kan=0;
  fillMyGraphReNorm(graphdEdxMeanVsBGV0KaonPos, hMeandEdxvsMomDataV0Kaon,   hSigmadEdxvsMomDataV0Kaon,   funcBBvsBGReference, 0.40, 2.0, iV0Kap);
  fillMyGraphReNorm(graphdEdxMeanVsBGV0KaonNeg, hMeandEdxvsMomDataV0KaonNeg,hSigmadEdxvsMomDataV0KaonNeg,funcBBvsBGReference, 0.40, 2.0, iV0Kan);
  
  //tpc-tof particles:
  TGraphErrors *graphdEdxMeanVsBGPionPos = new TGraphErrors();
  TGraphErrors *graphdEdxMeanVsBGKaonPos = new TGraphErrors();
  TGraphErrors *graphdEdxMeanVsBGProtPos = new TGraphErrors();
  TGraphErrors *graphdEdxMeanVsBGPionNeg = new TGraphErrors();
  TGraphErrors *graphdEdxMeanVsBGKaonNeg = new TGraphErrors();
  TGraphErrors *graphdEdxMeanVsBGProtNeg = new TGraphErrors();

  Int_t iTPCpip=0,iTPCpin=0,iTPCKap=0,iTPCKan=0,iTPCprp=0,iTPCprn=0;
  fillMyGraphReNorm(graphdEdxMeanVsBGPionPos, hMeandEdxvsMomDataPion,   hSigmadEdxvsMomDataPion,   funcBBvsBGReference, 1.4,  15.0, iTPCpip);
  fillMyGraphReNorm(graphdEdxMeanVsBGPionNeg, hMeandEdxvsMomDataPionNeg,hSigmadEdxvsMomDataPionNeg,funcBBvsBGReference, 1.4,  15.0, iTPCpin);  
  fillMyGraphReNorm(graphdEdxMeanVsBGKaonPos, hMeandEdxvsMomDataKaon,   hSigmadEdxvsMomDataKaon,   funcBBvsBGReference, 0.40,  2.0, iTPCKap);
  fillMyGraphReNorm(graphdEdxMeanVsBGKaonNeg, hMeandEdxvsMomDataKaonNeg,hSigmadEdxvsMomDataKaonNeg,funcBBvsBGReference, 0.40,  2.0, iTPCKan);
  fillMyGraphReNorm(graphdEdxMeanVsBGProtPos, hMeandEdxvsMomDataProt,   hSigmadEdxvsMomDataProt,   funcBBvsBGReference, 0.10,  2.0, iTPCprp);
  fillMyGraphReNorm(graphdEdxMeanVsBGProtNeg, hMeandEdxvsMomDataProtNeg,hSigmadEdxvsMomDataProtNeg,funcBBvsBGReference, 0.10,  2.0, iTPCprn);
  //------------------------------------------------------------------------------













  //// NOW FItting : 

  
  ROOT::Math::MinimizerOptions::SetDefaultMaxFunctionCalls(60000);

  TH1F *hEmpty = new TH1F("hEmpty","",5000,0.08,2E4);
  
  //// THis is the Canvas which shows actual graph that is fitted.
  TCanvas *Canvas = GetCanvas("cFitTGraph",xpos,ypos,600,460,0,0,0.02,0.16,0.12,0.01); // topMgn, botMgn, leftMgn, rightMgn
  Canvas->cd();
  Canvas->SetTicks();
  Canvas->SetLogx();
  Canvas->SetLogy();
  Canvas->SetGridy();
  //hEmpty->SetTitle("");
  SetTitleTH1(hEmpty,"#LT dE/dx #GT (Normalized)",0.06,0.95,"#beta#gamma",0.065,1.18);
  SetAxisTH1(hEmpty,21,1.2E4,0.21,2.4E4,0.06,0.06);
  hEmpty->Draw("");
  SetMarkerTH1(graphdEdxMeanVsBG,"",24,0.7,2,2);
  graphdEdxMeanVsBG->Draw("PSAME"); 
  graphdEdxMeanVsBG->Fit("funcBBvsBGThisPass","MREX0","I",0.008,2.1E4);  //// <====== HERE IS THE FITTING DONE ======
  //funcBBvsBGThisPass->SetLineColor(kGreen+2);
  funcBBvsBGThisPass->SetLineColor(2);
  
  double fchi2,fndf;
  fchi2 = funcBBvsBGThisPass->GetChisquare();
  fndf  = funcBBvsBGThisPass->GetNDF();
 
  legend = new TLegend(0.5,0.70,0.95,0.95);
  legend->SetBorderSize(0);
  legend->SetFillColor(0);
  legend->SetTextFont(42);
  legend->SetTextSize(0.045);
  legend->AddEntry(graphdEdxMeanVsBG,Form("%s",sDataSet.Data()),"");
  legend->AddEntry(graphdEdxMeanVsBG,"V0-daut: e^{#pm}, #pi^{#pm}, p#bar{p}","P");
  legend->AddEntry(graphdEdxMeanVsBG,"TPC-TOF: #pi^{#pm}, K^{#pm}, p#bar{p}","P");
  legend->AddEntry(funcBBvsBGThisPass,"BB Fit (Norm.Slice)","L");
  legend->AddEntry(funcBBvsBGThisPass,Form("Runs: %s",runFill.Data()),"");  
  legend->Draw(); 
  drawMyTextNDC(0.65, 0.455,0.04,Form("#chi^{2}/NDF: %3.1f / %3.0f",fchi2,fndf)); 
  if(Print)
    {
      std::string qaPath = CONFIG["output"]["fitBBGraph"]["QApath"].get<std::string>();
      TString qaPathT = TString::Format("%s", qaPath.c_str());
      if (!qaPathT.EndsWith("/")) qaPathT += "/";
      Canvas->SaveAs(Form("%sRealGraphFitdEdxvsBG_%s.pdf", qaPathT.Data(), sDataSet.Data()));
    }








  //return;




  //// THis following Canvas is for the Fancy figure with different species in different color:
  xpos+=150;
  TCanvas *Canvas2 = GetCanvas("cFitTGraphDiff",xpos,ypos,600,460,0,0,0.02,0.16,0.12,0.01); // topMgn, botMgn, leftMgn, rightMgn
  Canvas2->cd();
  Canvas2->SetTicks();
  Canvas2->SetLogx();
  Canvas2->SetLogy();
  Canvas2->SetGridy();
  hEmpty->Draw("");
  
  ///Only V0 selections:
  SetMarkerTH1(graphdEdxMeanVsBGV0ElecPos,"",30,0.95,1,1);
  graphdEdxMeanVsBGV0ElecPos->Draw("PSAMEX");
  SetMarkerTH1(graphdEdxMeanVsBGV0ElecNeg,"",30,0.8,kYellow-1,kYellow-1);
  graphdEdxMeanVsBGV0ElecNeg->Draw("PSAMEX");  
  SetMarkerTH1(graphdEdxMeanVsBGV0KaonPos,"",32,0.85,kOrange+7,kOrange+7);
  graphdEdxMeanVsBGV0KaonPos->Draw("PSAMEX");
  SetMarkerTH1(graphdEdxMeanVsBGV0KaonNeg,"",32,0.7,kOrange-3,kOrange-3);
  graphdEdxMeanVsBGV0KaonNeg->Draw("PSAMEX");

  ///V0 and TPC-TOF selections:
  SetMarkerTH1(graphdEdxMeanVsBGPionPos,"",24,0.75,kBlue+1,kBlue+1);
  graphdEdxMeanVsBGPionPos->Draw("PSAMEX");
  SetMarkerTH1(graphdEdxMeanVsBGPionNeg,"",24,0.6,kBlue-7,kBlue-7);
  graphdEdxMeanVsBGPionNeg->Draw("PSAMEX");
  SetMarkerTH1(graphdEdxMeanVsBGProtPos,"",25,0.65,kGreen+4,kGreen+4);
  graphdEdxMeanVsBGProtPos->Draw("PSAMEX");
  SetMarkerTH1(graphdEdxMeanVsBGProtNeg,"",25,0.5,kGreen-2,kGreen-2);
  graphdEdxMeanVsBGProtNeg->Draw("PSAMEX");
  
  ///Only TPC-TOF selections:
  SetMarkerTH1(graphdEdxMeanVsBGKaonPos,"",26,0.85,kMagenta+2,kMagenta+2);
  graphdEdxMeanVsBGKaonPos->Draw("PSAMEX");
  SetMarkerTH1(graphdEdxMeanVsBGKaonNeg,"",26,0.7,kMagenta-7,kMagenta-7);
  graphdEdxMeanVsBGKaonNeg->Draw("PSAMEX");
 
  ///Draw the BB fuction
  funcBBvsBGThisPass->Draw("LSAME");
  
  legend = new TLegend(0.62,0.715,0.95,0.95);
  legend->SetBorderSize(0);
  legend->SetFillColor(0);
  legend->SetTextFont(42);
  legend->SetTextSize(0.0425);
  legend->AddEntry(graphdEdxMeanVsBGV0ElecNeg,"","");
  legend->AddEntry(graphdEdxMeanVsBGV0ElecNeg,"e V0-daut","P");
  legend->AddEntry(graphdEdxMeanVsBGV0KaonNeg,"K V0-daut","P");
  legend->AddEntry(graphdEdxMeanVsBGPionNeg,"#pi V0,TPC-TOF","P");  
  legend->AddEntry(graphdEdxMeanVsBGProtNeg,"p V0, TPC-TOF","P");
  legend->AddEntry(graphdEdxMeanVsBGKaonNeg,"K TPC-TOF","P");
  legend->AddEntry(funcBBvsBGThisPass,"BB Fit","");
  legend->Draw();
  legend = new TLegend(0.68,0.715,0.95,0.95);
  legend->SetBorderSize(0);
  legend->SetFillColor(0);
  legend->SetTextFont(42);
  legend->SetTextSize(0.0425);
  legend->AddEntry(graphdEdxMeanVsBGV0ElecPos,Form("%s",sDataSet.Data()),"");
  legend->AddEntry(graphdEdxMeanVsBGV0ElecPos,"e V0-daut","P");
  legend->AddEntry(graphdEdxMeanVsBGV0KaonPos,"K V0-daut","P");
  legend->AddEntry(graphdEdxMeanVsBGPionPos,"#pi V0,TPC-TOF","P");  
  legend->AddEntry(graphdEdxMeanVsBGProtPos,"p V0, TPC-TOF","P");
  legend->AddEntry(graphdEdxMeanVsBGKaonPos,"K TPC-TOF","P");
  legend->AddEntry(funcBBvsBGThisPass,"BB Fit(Norm.Slice)","L");
  legend->Draw();
  

  

  drawMyTextNDC(0.642,0.91,0.0345,"-ve   +ve");  
  drawMyTextNDC(0.65, 0.45,0.04,Form("#chi^{2}/NDF: %3.1f / %3.0f",fchi2,fndf));
  drawMyTextNDC(0.625, 0.680,0.04,Form("Runs: %s",runFill.Data()));

  if(Print) 
    {
      std::string qaPath = CONFIG["output"]["fitBBGraph"]["QApath"].get<std::string>();
      TString qaPathT = TString::Format("%s", qaPath.c_str());
      if (!qaPathT.EndsWith("/")) qaPathT += "/";
      Canvas2->SaveAs(Form("%sGraphFitdEdxvsBG_%s.pdf", qaPathT.Data(), sDataSet.Data()));
    }
  
    
  
  Double_t ScorePion[2] = {0,};
  Double_t ScoreKaon[2] = {0,};
  Double_t ScoreProt[2] = {0,};

  Double_t ScoreElecV0[2] = {0,};
  Double_t ScorePionV0[2] = {0,};
  Double_t ScoreProtV0[2] = {0,};
  Double_t ScoreKaonV0[2] = {0,};

  
  ///Ratios with TPC-TOF:
  RenormHistogram(hMeandEdxvsMomDataPion, funcBBvsBGReference);
  RenormHistogram(hMeandEdxvsMomDataPionNeg, funcBBvsBGReference); 
  RenormHistogram(hMeandEdxvsMomDataKaon, funcBBvsBGReference);
  RenormHistogram(hMeandEdxvsMomDataKaonNeg, funcBBvsBGReference); 
  RenormHistogram(hMeandEdxvsMomDataProt, funcBBvsBGReference);
  RenormHistogram(hMeandEdxvsMomDataProtNeg, funcBBvsBGReference); 

  RenormHistogram(hMeandEdxvsMomDataV0Pion, funcBBvsBGReference);
  RenormHistogram(hMeandEdxvsMomDataV0PionNeg, funcBBvsBGReference);
  RenormHistogram(hMeandEdxvsMomDataV0Prot, funcBBvsBGReference);
  RenormHistogram(hMeandEdxvsMomDataV0ProtNeg, funcBBvsBGReference);
  RenormHistogram(hMeandEdxvsMomDataV0Elec, funcBBvsBGReference);
  RenormHistogram(hMeandEdxvsMomDataV0ElecNeg, funcBBvsBGReference); 
  RenormHistogram(hMeandEdxvsMomDataV0Kaon, funcBBvsBGReference);
  RenormHistogram(hMeandEdxvsMomDataV0KaonNeg, funcBBvsBGReference); 
  

  //plotPID2D(hdEdxvsMomDataPion, hMeandEdxvsMomDataPion, hMeandEdxvsMomDataPionNeg, "TPCToF_Pions", 0.140, 24, EColor::kRed, sDataSet,funcBBvsBGDefault,funcBBvsBGThisPass,ScorePion, 22,240,Print); // this function shows only tpc-tof particles canvas.
  //plotPID2D(hdEdxvsMomDataProt, hMeandEdxvsMomDataProt, hMeandEdxvsMomDataProtNeg, "TPCToF_ppbar", 0.938, 24, EColor::kRed, sDataSet,funcBBvsBGDefault,funcBBvsBGThisPass, ScoreProt,22,4.8E3,Print); // this function shows only tpc-tof particles in canvas.

  xpos+=150;
  plotPID2DMerged(hdEdxDataPionPosMerged, hMeandEdxvsMomDataPion, hMeandEdxvsMomDataPionNeg, hMeandEdxvsMomDataV0Pion, hMeandEdxvsMomDataV0PionNeg, "All_Pions_", 0.140, 24, EColor::kRed, sDataSet,funcBBvsBGDefault,funcBBvsBGThisPass,ScorePion, 22,240,Print); // this function shows merged tpc-tof and V0 in canvas.

  xpos+=150;
  plotPID2D(hdEdxvsMomDataKaon, hMeandEdxvsMomDataKaon, hMeandEdxvsMomDataKaonNeg, "TPCToF_Kaons_", 0.495, 24,  EColor::kRed, sDataSet,funcBBvsBGDefault,funcBBvsBGThisPass, ScoreKaon, 22,2.4E3,Print);  

  xpos+=150;
  plotPID2DMerged(hdEdxDataProtPosMerged, hMeandEdxvsMomDataProt, hMeandEdxvsMomDataProtNeg, hMeandEdxvsMomDataV0Prot, hMeandEdxvsMomDataV0ProtNeg, "All_ppbar_", 0.938, 24, EColor::kRed, sDataSet,funcBBvsBGDefault,funcBBvsBGThisPass, ScoreProt,22,4.8E3,Print); // this function shows merged tpc-tof and V0 in canvas.
   
  xpos+=150;  
  plotPID2D(hdEdxvsMomDataV0Elec, hMeandEdxvsMomDataV0Elec, hMeandEdxvsMomDataV0ElecNeg, "V0_elec_", 0.000511, 24, EColor::kRed, sDataSet,funcBBvsBGDefault,funcBBvsBGThisPass, ScoreElecV0, 34,240,Print);

  xpos+=150;  
  plotPID2D(hdEdxvsMomDataV0Kaon, hMeandEdxvsMomDataV0Kaon, hMeandEdxvsMomDataV0KaonNeg, "V0_Kaon_", 0.495, 24, EColor::kRed, sDataSet,funcBBvsBGDefault,funcBBvsBGThisPass, ScoreKaonV0, 22,2.4E3,Print);




  
  /// We do not need to plot the V0 Pion and V0 protons separately,
  // As we have already plotted them above, along with tpc-tof. 
  /*
  /// Now Ratios with V0:
  xpos+=150;
  plotPID2D(hdEdxvsMomDataV0Pion, hMeandEdxvsMomDataV0Pion, hMeandEdxvsMomDataV0PionNeg, "V0_Pions", 0.140, 24, EColor::kRed, sDataSet,funcBBvsBGDefault,funcBBvsBGThisPass, ScorePionV0, 22,240,Print);
  xpos+=150;
  plotPID2D(hdEdxvsMomDataV0Prot, hMeandEdxvsMomDataV0Prot, hMeandEdxvsMomDataV0ProtNeg, "V0_ppbar", 0.938, 24, EColor::kRed, sDataSet,funcBBvsBGDefault,funcBBvsBGThisPass, ScoreProtV0, 22,4.2E3,Print);
  */


  Double_t sumScoreTGF = sqrt(ScorePion[0]*ScorePion[0] + ScoreKaon[0]*ScoreKaon[0] + ScoreProt[0]*ScoreProt[0] +
		      ScoreElecV0[0]*ScoreElecV0[0] + ScorePionV0[0]*ScorePionV0[0] + ScoreProtV0[0]*ScoreProtV0[0] + ScoreKaonV0[0]*ScoreKaonV0[0]);

  Double_t sumScoreHPO = sqrt(ScorePion[1]*ScorePion[1] + ScoreKaon[1]*ScoreKaon[1] + ScoreProt[1]*ScoreProt[1] +
		      ScoreElecV0[1]*ScoreElecV0[1] + ScorePionV0[1]*ScorePionV0[1] + ScoreProtV0[1]*ScoreProtV0[1] + ScoreKaonV0[1]*ScoreKaonV0[1]);
    
  cout<<" Total Score (All Species) from new BB (TGraph) Fit: "<<sumScoreTGF<<","<<endl;
  cout<<" Total Score (All Species) from Default Old BB par: "<<sumScoreHPO<<" (period)."<<endl;
 
  Double_t params[6] = {0,};
  funcBBvsBGThisPass->GetParameters(params);

  CONFIG["output"]["fitBBGraph"]["BBparameters"] = nlohmann::json::array();
  for (int i = 0; i < 5; i++) {
    CONFIG["output"]["fitBBGraph"]["BBparameters"].push_back(params[i]);
  }
  writeConfig();


  
}//main ends










//// Use Function to do same boring stuff again and again:


/// Graph Without Error:

void fillMyGraph(TGraphErrors *g1, TH1* hInput, int &count){

  if(!g1 || !hInput){
    cout<<"fillMyGraph w/o range: TGraphError pointer or Input Histogram is missing! \n Exit! "<<endl;
  }
    
  int NbinX = hInput->GetNbinsX();
  double binCent,binCont, binErr, binWidth;
    
  for(int i=1; i<=NbinX; i++){
    binCont = hInput->GetBinContent(i);
    if(binCont<0) continue; 
    binCent = hInput->GetBinCenter(i);
    binErr  = hInput->GetBinError(i);
    binWidth= hInput->GetBinWidth(i);
    //cout<<"indx:"<<count<<" binCent: "<<binCent<<"\t cont: "<<binCont<<"\tErr:"<<binErr<<endl;
    if(binErr/binCont > 0.4) continue;
    
    g1->SetPoint(count,binCent,binCont);
    g1->SetPointError(count,binWidth/2.0,binErr);
    //g1->SetPointError(count,0.1*binWidth/2.0,0.10);
    //g1->SetPointError(count,0.0,binErr);
    count++;
  }
}

void fillMyGraph(TGraphErrors *g1, TH1* hInput, double xLow=0.2, double xHigh=2E4, int &count){

  if(!g1 || !hInput){
    cout<<" fillMyGraph w/ range : Either TGraphError pointer or Input Histogram is missing! \n Exit! "<<endl;
  }
    
  int NbinX = hInput->GetNbinsX();
  double binCent,binCont, binErr, binWidth;
    
  for(int i=1; i<=NbinX; i++){
    binCont = hInput->GetBinContent(i);
    if(binCont<0) continue; 
    binCent = hInput->GetBinCenter(i);
    binErr  = hInput->GetBinError(i);
    binWidth= hInput->GetBinWidth(i);
    //cout<<"indx:"<<count<<" binCent: "<<binCent<<"\t cont: "<<binCont<<"\tErr:"<<binErr<<endl;
    if(binErr/binCont > 0.4) continue;
    if(binCent < xLow || binCent > xHigh) continue;
    
    g1->SetPoint(count,binCent,binCont);
    g1->SetPointError(count,binWidth/2.0,0.0);
    //g1->SetPointError(count,0.1*binWidth/2.0,0.10);
    //g1->SetPointError(count,0.0,binErr);
    count++;
  }
}

void RenormHistogram(TH1 *hInput, TF1 *funcBB){

  if(!hInput || !funcBB){
    cout<<"\n RenormHistogram WARNING::: Either Histogram or TF1 pointer is missing! \n Exit! "<<endl;
  }

  int NbinX = hInput->GetNbinsX();
  double binCent,binCont, binErr, binWidth, refdEdx;
  
  for(int i=1; i<=NbinX; i++){
    binCont = hInput->GetBinContent(i);
    if(binCont<0) continue; 
    binCent = hInput->GetBinCenter(i);
    binErr  = hInput->GetBinError(i);
    binWidth= hInput->GetBinWidth(i);
    //cout<<" binCent: "<<binCent<<"\t cont: "<<binCont<<"\tErr:"<<binErr<<endl;
    if(binErr/binCont > 0.4) continue;
    //if(binCent < xLow || binCent > xHigh) continue;
    refdEdx = funcBB->Eval(binCent);
    
    hInput->SetBinContent(i,binCont*refdEdx);
    hInput->SetBinError(i,binErr*refdEdx);
    //g1->SetPointError(count,0.1*binWidth/2.0,0.10);
    //g1->SetPointError(count,0.0,binErr);
    //count++;
  }
  
}



/// This Function is filling the graph that is fitted with BB function:

void fillMyGraphReNorm(TGraphErrors *g1, TH1* hInput, TF1* funcBB, double xLow, double xHigh, int &count){

  if(!g1 || !hInput){
    cout<<"fillMyGraphReNorm w/o error: TGraphError pointer or Input Histogram is missing! \n Exit! "<<endl;
  }
    
  int NbinX = hInput->GetNbinsX();
  double binCent,binCont, binErr, binWidth, refdEdx, newdEdx;
    
  for(int i=1; i<=NbinX; i++){
    binCont = hInput->GetBinContent(i);
    if(binCont<0) continue; 
    binCent = hInput->GetBinCenter(i);
    binErr  = hInput->GetBinError(i);
    binWidth= hInput->GetBinWidth(i);
    //cout<<"indx:"<<count<<" binCent: "<<binCent<<"\t cont: "<<binCont<<"\tErr:"<<binErr<<endl;
    if(binErr/binCont > 0.4) continue;
    if(binCent < xLow || binCent > xHigh) continue;
    refdEdx = funcBB->Eval(binCent);
    newdEdx = binCont*refdEdx;
    //condition for electrons:
    if(binCent>500){
      cout<<"func1 binCent: "<<binCent<<" newdEdx: "<<newdEdx<<" refBB: "<<endl;
      if(newdEdx/refdEdx > 1.1 || newdEdx/refdEdx < 0.90)
	continue;
    }
    else if(binCent<5){
      if(newdEdx/refdEdx > 1.10 || newdEdx/refdEdx < 0.90)
	continue;
    }

	    
    
    g1->SetPoint(count,binCent,newdEdx);
    g1->SetPointError(count,binWidth/2.0, newdEdx*0.02);
    //g1->SetPointError(count,0.1*binWidth/2.0,0.10);
    //g1->SetPointError(count,0.0,binErr);
    count++;
  }
}





										 
void fillMyGraphReNorm(TGraphErrors *g1, TH1* hInput, TH1* hError, TF1* funcBB, double xLow, double xHigh, int &count){
  
  if(!g1 || !hInput){
    cout<<" fillMyGraphReNorm w.error: TGraphError pointer or Input Histogram is missing! \n Exit! "<<endl;
  }
    
  int NbinX = hInput->GetNbinsX();
  double binCent,binCont, binErr, binWidth,refdEdx,newdEdx;
    
  for(int i=1; i<=NbinX; i++){
    binCont = hInput->GetBinContent(i);
    if(binCont<0) continue; 
    binCent = hInput->GetBinCenter(i);
    if(binCent < xLow || binCent > xHigh) continue;
    binErr  = hError->GetBinContent(i);
    binWidth= hInput->GetBinWidth(i);
    //cout<<"indx:"<<count<<" binCent: "<<binCent<<"\t cont: "<<binCont<<"\tErr:"<<binErr<<endl;
    if(binErr/binCont > 0.4) continue;

    refdEdx = funcBB->Eval(binCent);

    newdEdx = binCont*refdEdx;
    //condition for electrons:
    if(binCent>500){
      cout<<"func2 binCent: "<<binCent<<" newdEdx: "<<newdEdx<<" refBB: "<<refdEdx<<endl;
      if(newdEdx/refdEdx > 1.1 || newdEdx/refdEdx < 0.90)
	continue;
    }
    else if(binCent<5){
      if(newdEdx/refdEdx > 1.10 || newdEdx/refdEdx < 0.90)
	continue;
    }
    
    g1->SetPoint(count,binCent,newdEdx);
    //g1->SetPointError(count,0.1*binWidth/2.0,binErr);
    //g1->SetPointError(count,0.1*binWidth/2.0,0.10);
    g1->SetPointError(count,binWidth/2.0,newdEdx);
    count++;
  }
}





void fillMyGraph(TGraphErrors *g1, TH1* hInput, TH1* hError, double xLow, double xHigh, int &count){
  
  if(!g1 || !hInput){
    cout<<" Either TGraphError pointer or Input Histogram is missing! \n Exit! "<<endl;
  }
    
  int NbinX = hInput->GetNbinsX();
  double binCent,binCont, binErr, binWidth;
    
  for(int i=1; i<=NbinX; i++){
    binCont = hInput->GetBinContent(i);
    if(binCont<0) continue; 
    binCent = hInput->GetBinCenter(i);
    if(binCent < xLow || binCent > xHigh) continue;
    binErr  = hError->GetBinContent(i);
    binWidth= hInput->GetBinWidth(i);
    //cout<<"indx:"<<count<<" binCent: "<<binCent<<"\t cont: "<<binCont<<"\tErr:"<<binErr<<endl;
    if(binErr/binCont > 0.5) continue;
    
    g1->SetPoint(count,binCent,binCont);
    //g1->SetPointError(count,0.1*binWidth/2.0,binErr);
    //g1->SetPointError(count,0.1*binWidth/2.0,0.10);
    g1->SetPointError(count,binWidth/2.0,binErr);
    count++;
  }
}


//// FILL GRAPH WITH ERROR:
void fillMyGraph(TGraphErrors *g1, TH1* hInput,TH1* hError, int &count){

  if(!g1 || !hInput){
    cout<<" Either TGraphError pointer or Input Histogram is missing! \n Exit! "<<endl;
  }
    
  int NbinX = hInput->GetNbinsX();
  double binCent,binCont, binErr, binWidth;
    
  for(int i=1; i<=NbinX; i++){
    binCont = hInput->GetBinContent(i);
    if(binCont<0) continue; 
    binCent = hInput->GetBinCenter(i);
    binErr  = hError->GetBinContent(i);
    binWidth= hInput->GetBinWidth(i);
    //cout<<"indx:"<<count<<" binCent: "<<binCent<<"\t cont: "<<binCont<<"\tErr:"<<binErr<<endl;
    if(binErr/binCont > 0.5) continue;
    
    g1->SetPoint(count,binCent,binCont);
    //g1->SetPointError(count,0.1*binWidth/2.0,binErr);
    //g1->SetPointError(count,0.1*binWidth/2.0,0.10);
    g1->SetPointError(count,binWidth/2.0,binErr);
    count++;
  }
}


void plotPID2DMerged(TH2 *hdEdx2Ddata, TH1 *hdEdxMeanData, TH1 *hdEdxMeanDataNeg, TH1 *hdEdxMeanDataV0=0x0, TH1 *hdEdxMeanDataV0Neg=0x0, TString name, Float_t fMass, int iMarker, int ci, TString passName, TF1 *funcPass3, TF1* funcPass2, Double_t *scores, Float_t ySLow, Float_t ySHigh, Bool_t Print){

  TLine *line;
  TLatex  *tex;
  TLegend *legend;
  Double_t yLblPos = 0.825, xLblPos = 0.85;
  Double_t xSLow = 0.10, xSHigh=110;
  
  ///Different Range for Electron:

  if(name.Contains("elec") || name.Contains("Elec")){
    xSLow = 90;
    xSHigh=2E4;
  }
  if(name.Contains("Pion")){
    xSLow=0.31;
    xSHigh=110;
  }
    
  //if(!strncmp(name,"ppbar",4)){
    //xSLow = 0.1014, xSHigh=1800;
  //}
  ///0.101176, 0.101383, 0.101626, 0.101913, 0.102249, 0.102645, 0.103111, 
  //TH1D *hdEdxMeanData  = hp->ProjectionX(Form("hdEdxMeanData%s",name.Data()),"e");
  
  TCanvas *Canvas = GetCanvas(Form("CdEdxBG%s",name.Data()),xpos,ypos,cansizeX,cansizeY,0,0,0.02,0.02,0.14,0.01);
  Canvas->cd();
  //Func: GetPad(name.Data(), xpos1, ypos1, xpos2, ypos2, topMar, botMar, leftMar, rightMar)
  TPad *padTop = GetPad(Form("padTop%s",name.Data()),0.0,0.401,1.0,1.0, 0.02,0.0,0.14,0.01);
  padTop->Draw();
  padTop->cd();
  padTop->SetTicks();
  padTop->SetLogx();
  padTop->SetLogy();
  padTop->SetLogz();
  padTop->SetGridy();


  hdEdx2Ddata->SetTitle("");
  SetTitleTH1(hdEdx2Ddata," dE/dx ",0.08,0.8,"#beta#gamma",0.06,xLblPos);
  SetAxisTH1(hdEdx2Ddata,ySLow,ySHigh,xSLow,xSHigh,0.06,0.06);
  SetAxisTH1(hdEdx2Ddata,ySLow,ySHigh,xSLow,xSHigh,0.06,0.06);
  //SetMarkerTH1(hdEdx2Ddata,"",iMarker,0.8,ci,ci);
  hdEdx2Ddata->Draw("COLZ");

  SetMarkerTH1(hdEdxMeanData,"",iMarker,0.8,kRed,kRed);  
  hdEdxMeanData->Draw("PSAME");
  SetMarkerTH1(hdEdxMeanDataV0,"",25,1.0,kMagenta+1,kMagenta+1); 
  hdEdxMeanDataV0->Draw("PSAME");

  funcPass2->SetRange(xSLow,9E4);
  funcPass3->SetRange(xSLow,9E4);
  
  funcPass2->SetLineColor(12);
  funcPass2->Draw("LSAME");
  funcPass3->SetLineColor(kGreen+2);
  funcPass3->Draw("LSAME");
  

  legend = new TLegend(0.55,0.675,0.95,0.95);
  legend->SetBorderSize(0);
  legend->SetFillColor(0);
  legend->SetTextFont(42);
  legend->SetTextSize(0.045);
  legend->AddEntry(hdEdxMeanData,Form("%s",passName.Data()),"");
  legend->AddEntry(hdEdxMeanData,Form("%s",name.Data()),"");
  legend->AddEntry(hdEdxMeanData,"tpc-tof","P");
  legend->AddEntry(hdEdxMeanDataV0,"V0","P");  
  legend->AddEntry(funcPass2,"New-Fit,(Norm.)","L");
  legend->AddEntry(funcPass3,"Default","L");
  legend->Draw();

  Canvas->Update();
  Canvas->cd();
  TPad *padBott = GetPad(Form("padBott%s",name.Data()),0.0,0.0,1.0,0.400, 0.02,0.24,0.14,0.01);
  padBott->Draw();
  padBott->cd();
  padBott->SetLogx();
  padBott->SetTicks();
  padBott->SetGridy();

  ///Ratio w/ pass2 
  TH1D *hdEdxMeanRatioPass2 = (TH1D *) hdEdxMeanData->Clone(Form("hdEdxMeanRatioPass2%s",name.Data()));
  hdEdxMeanRatioPass2->Divide(funcPass2,1.);
  TH1D *hdEdxMeanRatioPass2Neg = (TH1D *) hdEdxMeanDataNeg->Clone(Form("hdEdxMeanRatioPass2Neg%s",name.Data()));
  hdEdxMeanRatioPass2Neg->Divide(funcPass2,1.);
  
  TH1D *hdEdxMeanRatioPass2V0 = (TH1D *) hdEdxMeanDataV0->Clone(Form("hdEdxMeanRatioPass2V0%s",name.Data()));
  hdEdxMeanRatioPass2V0->Divide(funcPass2,1.);
  TH1D *hdEdxMeanRatioPass2V0Neg = (TH1D *) hdEdxMeanDataV0Neg->Clone(Form("hdEdxMeanRatioPass2V0Neg%s",name.Data()));
  hdEdxMeanRatioPass2V0Neg->Divide(funcPass2,1.);
  
  

  /// Get the Score for GraphFit:
  Double_t scGrPos=0,scGrNeg=0, scGrTot=0;
  scGrPos = GetScoreRatio(hdEdxMeanRatioPass2) + GetScoreRatio(hdEdxMeanRatioPass2V0);
  scGrNeg = GetScoreRatio(hdEdxMeanRatioPass2Neg)+GetScoreRatio(hdEdxMeanRatioPass2V0Neg);
  scGrTot = sqrt(scGrPos*scGrPos + scGrNeg*scGrNeg);
  cout<<"\n GraphFit Score for "<<name.Data()<<"\t: Pos:"<<scGrPos<<"\t Neg: "<<scGrNeg<<"\t Tot: "<<scGrTot<<endl;
  scores[0] = scGrTot;
  
  
  ///Ratio w/ pass3 
  TH1D *hdEdxMeanRatioPass3 = (TH1D *) hdEdxMeanData->Clone(Form("hdEdxMeanRatioPass3%s",name.Data()));
  hdEdxMeanRatioPass3->Divide(funcPass3,1.);
  TH1D *hdEdxMeanRatioPass3Neg = (TH1D *) hdEdxMeanDataNeg->Clone(Form("hdEdxMeanRatioPass3Neg%s",name.Data()));
  hdEdxMeanRatioPass3Neg->Divide(funcPass3,1.);
  
  TH1D *hdEdxMeanRatioPass3V0 = (TH1D *) hdEdxMeanDataV0->Clone(Form("hdEdxMeanRatioPass3V0%s",name.Data()));
  hdEdxMeanRatioPass3V0->Divide(funcPass3,1.);
  TH1D *hdEdxMeanRatioPass3V0Neg = (TH1D *) hdEdxMeanDataV0Neg->Clone(Form("hdEdxMeanRatioPass3V0Neg%s",name.Data()));
  hdEdxMeanRatioPass3V0Neg->Divide(funcPass3,1.);
  
  
  ///------ Get the Score for Hyper Param Optimization: -------------
  Double_t scHPPos=0,scHPNeg=0, scHPTot=0;
  scHPPos = GetScoreRatio(hdEdxMeanRatioPass3) + GetScoreRatio(hdEdxMeanRatioPass3V0);
  scHPNeg = GetScoreRatio(hdEdxMeanRatioPass3Neg) + GetScoreRatio(hdEdxMeanRatioPass3V0Neg);
  scHPTot = sqrt(scHPPos*scHPPos + scHPNeg*scHPNeg);
  cout<<"Hyp.Optim. Score for "<<name.Data()<<"\t: Pos:"<<scHPPos<<"\t Neg: "<<scHPNeg<<"\t Tot: "<<scHPTot<<endl;
  //-----------------------------------------------------------------
  
  
  scores[1] = scHPTot;
  //hdEdxMeanRatioPass3->Divide(hdEdxMeanRatioPass3,hdEdxTheoPass2,1.,1.,"B");

  hdEdxMeanRatioPass2->GetYaxis()->SetNdivisions(507);				
  SetTitleTH1(hdEdxMeanRatioPass2,"Data / BB ",0.10,0.65,"#beta#gamma",0.10,1.0);
  SetAxisTH1(hdEdxMeanRatioPass2,0.8614,1.1525,xSLow,xSHigh,0.08,0.08);
  SetMarkerTH1(hdEdxMeanRatioPass2,"",2,0.8,kRed,kRed);
  hdEdxMeanRatioPass2->Draw("P");
  SetMarkerTH1(hdEdxMeanRatioPass2Neg,"",2,0.8,kBlue,kBlue);
  hdEdxMeanRatioPass2Neg->Draw("PSAME");
  SetMarkerTH1(hdEdxMeanRatioPass2V0,"",5,0.8,kRed+2,kRed+2);
  hdEdxMeanRatioPass2V0->Draw("PSAME");
  SetMarkerTH1(hdEdxMeanRatioPass2V0Neg,"",5,0.8,kBlue+2,kBlue+2);
  hdEdxMeanRatioPass2V0Neg->Draw("PSAME");
  /*
  SetMarkerTH1(hdEdxMeanRatioPass3,"",iMarker,0.75,kRed,kRed);
  hdEdxMeanRatioPass3->Draw("PSAME");
  SetMarkerTH1(hdEdxMeanRatioPass3Neg,"",iMarker,0.75,kBlue,kBlue);
  hdEdxMeanRatioPass3Neg->Draw("PSAME");
  SetMarkerTH1(hdEdxMeanRatioPass3V0,"",25,0.85,kRed+2,kRed+2);
  hdEdxMeanRatioPass3V0->Draw("PSAME");
  SetMarkerTH1(hdEdxMeanRatioPass3V0Neg,"",25,0.85,kBlue+2,kBlue+2);
  hdEdxMeanRatioPass3V0Neg->Draw("PSAME");
  */

  //drawMyTextNDC(0.1, 0.085,0.07,Form("GRFitScore:  %1.3f",scGrTot));
  //drawMyTextNDC(0.1, 0.025,0.07,Form("HPOptScore: %1.3f",scHPTot));
 
  legend = new TLegend(0.35,0.25,0.575,0.40);
  legend->SetBorderSize(0);
  legend->SetFillColor(0);
  legend->SetTextFont(42);
  legend->SetTextSize(0.056);
  legend->AddEntry(hdEdxMeanRatioPass2,"PosTPC/NewFit","P");
  legend->AddEntry(hdEdxMeanRatioPass2Neg,"NegTPC/NewFit","P");
  legend->Draw();

  legend = new TLegend(0.595,0.25,0.75,0.40);
  legend->SetBorderSize(0);
  legend->SetFillColor(0);
  legend->SetTextFont(42);
  legend->SetTextSize(0.056);
  legend->AddEntry(hdEdxMeanRatioPass2V0,"PosV0/NewFit","P");
  legend->AddEntry(hdEdxMeanRatioPass2V0Neg,"NegV0/NewFit","P");
  legend->Draw();
  
  ///--------- Marking in momentum --------
  Float_t xBGLow = 0.2/fMass;
  Float_t xBGHigh= 7.0/fMass;
  Float_t yLinePos= 1.12; //10*funcPass2->Eval(1.0/fMass);


  ///Momentum Line, and Start and End ticks:
  drawMyline(xBGLow,yLinePos,xBGHigh,yLinePos, 1, 2, 1);
  drawMyline(xBGLow,yLinePos,xBGLow,yLinePos-0.02, 1, 2, 1);
  drawMyline(xBGHigh,yLinePos,xBGHigh,yLinePos-0.02, 1, 2, 1);

  
  tex = new TLatex(xBGLow*0.9,yLinePos+0.005,"0.2");
  tex->SetTextFont(42);
  tex->SetTextSize(0.06);
  tex->SetLineWidth(2);
  tex->Draw();

  tex = new TLatex(xBGLow*1.1,yLinePos-0.045,"p (GeV/c)");
  tex->SetTextFont(42);
  tex->SetTextSize(0.06);
  tex->SetLineWidth(2);
  tex->Draw();
  
  tex = new TLatex(xBGHigh*0.95,yLinePos+0.005,"7");
  tex->SetTextFont(42);
  tex->SetTextSize(0.06);
  tex->SetLineWidth(2);
  tex->Draw();
  
  //------Now Middle ticks:   
  xBGHigh= 0.5/fMass;
  drawMyline(xBGHigh,yLinePos,xBGHigh,yLinePos-0.02, 1, 2, 1);  
  tex = new TLatex(xBGHigh*0.95,yLinePos+0.005,"0.5");
  tex->SetTextFont(42);
  tex->SetTextSize(0.06);
  tex->SetLineWidth(2);
  tex->Draw();

  xBGHigh= 1.0/fMass;
  drawMyline(xBGHigh,yLinePos,xBGHigh,yLinePos-0.02, 1, 2, 1);  
  tex = new TLatex(xBGHigh*0.95,yLinePos+0.005,"1");
  tex->SetTextFont(42);
  tex->SetTextSize(0.06);
  tex->SetLineWidth(2);
  tex->Draw();

  xBGHigh= 2.0/fMass;
  drawMyline(xBGHigh,yLinePos,xBGHigh,yLinePos-0.02, 1, 2, 1);  
  tex = new TLatex(xBGHigh*0.95,yLinePos+0.005,"2");
  tex->SetTextFont(42);
  tex->SetTextSize(0.06);
  tex->SetLineWidth(2);
  tex->Draw();
  
  xBGHigh= 4.0/fMass;
  drawMyline(xBGHigh,yLinePos,xBGHigh,yLinePos-0.02, 1, 2, 1);  
  tex = new TLatex(xBGHigh*0.95,yLinePos+0.005,"4");
  tex->SetTextFont(42);
  tex->SetTextSize(0.06);
  tex->SetLineWidth(2);
  tex->Draw();
  //---------------------------
  
  
  
  if(Print)
    {
      std::string qaPath = CONFIG["output"]["fitBBGraph"]["QApath"].get<std::string>();
      TString qaPathT = TString::Format("%s", qaPath.c_str());
      if (!qaPathT.EndsWith("/")) qaPathT += "/";
      Canvas->SaveAs(Form("%sdEdxvsBetaGamma%s%s.pdf", qaPathT.Data(), name.Data(),passName.Data()));
    }
}






void plotPID2D(TH2 *hdEdx2Ddata, TH1 *hdEdxMeanData, TH1 *hdEdxMeanDataNeg, TString name, Float_t fMass, int iMarker, int ci, TString passName, TF1 *funcPass3, TF1* funcPass2, Double_t *scores, Float_t ySLow, Float_t ySHigh, Bool_t Print){

  TLine *line;
  TLatex  *tex;
  TLegend *legend;
  Double_t yLblPos = 0.825, xLblPos = 0.85;
  Double_t xSLow = 0.10, xSHigh=110;
  
  ///Different Range for Electron:

  if(name.Contains("elec") || name.Contains("Elec")){
    xSLow = 90;
    xSHigh=2E4;
  }
  if(name.Contains("Pion")){
    xSLow=0.31;
    xSHigh=110;
  }
    
  //if(!strncmp(name,"ppbar",4)){
    //xSLow = 0.1014, xSHigh=1800;
  //}
  ///0.101176, 0.101383, 0.101626, 0.101913, 0.102249, 0.102645, 0.103111, 
  //TH1D *hdEdxMeanData  = hp->ProjectionX(Form("hdEdxMeanData%s",name.Data()),"e");
  
  TCanvas *Canvas = GetCanvas(Form("CdEdxBG%s",name.Data()),xpos,ypos,cansizeX,cansizeY,0,0,0.02,0.02,0.14,0.01);
  Canvas->cd();
  //Func: GetPad(name.Data(), xpos1, ypos1, xpos2, ypos2, topMar, botMar, leftMar, rightMar)
  TPad *padTop = GetPad(Form("padTop%s",name.Data()),0.0,0.401,1.0,1.0, 0.02,0.0,0.14,0.01);
  padTop->Draw();
  padTop->cd();
  padTop->SetTicks();
  padTop->SetLogx();
  padTop->SetLogy();
  padTop->SetLogz();
  padTop->SetGridy();


  hdEdx2Ddata->SetTitle("");
  SetTitleTH1(hdEdx2Ddata," dE/dx ",0.08,0.8,"#beta#gamma",0.06,xLblPos);
  SetAxisTH1(hdEdx2Ddata,ySLow,ySHigh,xSLow,xSHigh,0.06,0.06);
  SetAxisTH1(hdEdx2Ddata,ySLow,ySHigh,xSLow,xSHigh,0.06,0.06);
  //SetMarkerTH1(hdEdx2Ddata,"",iMarker,0.8,ci,ci);
  hdEdx2Ddata->Draw("COLZ");
  /*
  SetTitleTH1(hdEdxMeanData,"#LT dE/dx #GT",0.08,0.8,"#beta#gamma",0.06,xLblPos);
  SetAxisTH1(hdEdxMeanData,ySLow,ySHigh,xSLow,xSHigh,0.07,0.07);
  SetAxisTH1(hdEdxMeanData,ySLow,ySHigh,xSLow,xSHigh,0.07,0.07);
  */
  SetMarkerTH1(hdEdxMeanData,"",iMarker,0.8,ci,ci);  
  hdEdxMeanData->Draw("PSAME");

  funcPass2->SetRange(xSLow,9E4);
  funcPass3->SetRange(xSLow,9E4);
  
  funcPass2->SetLineColor(12);
  funcPass2->Draw("LSAME");
  funcPass3->SetLineColor(kGreen+2);
  funcPass3->Draw("LSAME");
  

  legend = new TLegend(0.55,0.675,0.95,0.95);
  legend->SetBorderSize(0);
  legend->SetFillColor(0);
  legend->SetTextFont(42);
  legend->SetTextSize(0.0525);
  legend->AddEntry(hdEdxMeanData,Form("%s",passName.Data()),"");
  legend->AddEntry(hdEdxMeanData,Form("%s",name.Data()),"P");
  legend->AddEntry(funcPass2,"New-Fit,(Norm.)","L");
  legend->AddEntry(funcPass3,"Default","L");
  legend->Draw();

  Canvas->Update();
  Canvas->cd();
  TPad *padBott = GetPad(Form("padBott%s",name.Data()),0.0,0.0,1.0,0.400, 0.02,0.24,0.14,0.01);
  padBott->Draw();
  padBott->cd();
  padBott->SetLogx();
  padBott->SetTicks();
  padBott->SetGridy();

  ///Ratio w/ pass2 
  TH1D *hdEdxMeanRatioPass2 = (TH1D *) hdEdxMeanData->Clone(Form("hdEdxMeanRatioPass2%s",name.Data()));
  hdEdxMeanRatioPass2->Divide(funcPass2,1.);
  TH1D *hdEdxMeanRatioPass2Neg = (TH1D *) hdEdxMeanDataNeg->Clone(Form("hdEdxMeanRatioPass2Neg%s",name.Data()));
  hdEdxMeanRatioPass2Neg->Divide(funcPass2,1.);

  /// Get the Score for GraphFit:
  Double_t scGrPos=0,scGrNeg=0, scGrTot=0;
  scGrPos = GetScoreRatio(hdEdxMeanRatioPass2);
  scGrNeg = GetScoreRatio(hdEdxMeanRatioPass2Neg);
  scGrTot = sqrt(scGrPos*scGrPos + scGrNeg*scGrNeg);
  cout<<"\n GraphFit Score for "<<name.Data()<<"\t: Pos:"<<scGrPos<<"\t Neg: "<<scGrNeg<<"\t Tot: "<<scGrTot<<endl;
  scores[0] = scGrTot;
  
  
  ///Ratio w/ pass3 
  TH1D *hdEdxMeanRatioPass3 = (TH1D *) hdEdxMeanData->Clone(Form("hdEdxMeanRatioPass3%s",name.Data()));
  hdEdxMeanRatioPass3->Divide(funcPass3,1.);
  TH1D *hdEdxMeanRatioPass3Neg = (TH1D *) hdEdxMeanDataNeg->Clone(Form("hdEdxMeanRatioPass3Neg%s",name.Data()));
  hdEdxMeanRatioPass3Neg->Divide(funcPass3,1.);

  /// Get the Score for Hyper Param Optimization:
  Double_t scHPPos=0,scHPNeg=0, scHPTot=0;
  scHPPos = GetScoreRatio(hdEdxMeanRatioPass3);
  scHPNeg = GetScoreRatio(hdEdxMeanRatioPass3Neg);
  scHPTot = sqrt(scHPPos*scHPPos + scHPNeg*scHPNeg);
  cout<<"Hyp.Optim. Score for "<<name.Data()<<"\t: Pos:"<<scHPPos<<"\t Neg: "<<scHPNeg<<"\t Tot: "<<scHPTot<<endl;

  scores[1] = scHPTot;
  //hdEdxMeanRatioPass3->Divide(hdEdxMeanRatioPass3,hdEdxTheoPass2,1.,1.,"B");

  hdEdxMeanRatioPass2->GetYaxis()->SetNdivisions(507);				
  SetTitleTH1(hdEdxMeanRatioPass2,"Data / BB ",0.10,0.65,"#beta#gamma",0.10,1.0);
  SetAxisTH1(hdEdxMeanRatioPass2,0.8614,1.1525,xSLow,xSHigh,0.08,0.08);
  SetMarkerTH1(hdEdxMeanRatioPass2,"",2,0.8,kRed+2,kRed+2);
  hdEdxMeanRatioPass2->Draw("P");
  SetMarkerTH1(hdEdxMeanRatioPass2Neg,"",2,0.8,kBlue+2,kBlue+2);
  hdEdxMeanRatioPass2Neg->Draw("PSAME");
  
  SetMarkerTH1(hdEdxMeanRatioPass3,"",iMarker,0.75,ci,ci);
  hdEdxMeanRatioPass3->Draw("PSAME");
  SetMarkerTH1(hdEdxMeanRatioPass3Neg,"",iMarker,0.75,kBlue,kBlue);
  hdEdxMeanRatioPass3Neg->Draw("PSAME");


  //drawMyTextNDC(0.1, 0.085,0.07,Form("GRFitScore:  %1.3f",scGrTot));
  //drawMyTextNDC(0.1, 0.025,0.07,Form("HPOptScore: %1.3f",scHPTot));
 
  legend = new TLegend(0.35,0.25,0.575,0.40);
  legend->SetBorderSize(0);
  legend->SetFillColor(0);
  legend->SetTextFont(42);
  legend->SetTextSize(0.056);
  legend->AddEntry(hdEdxMeanRatioPass3,"Pos/Deft","P");
  legend->AddEntry(hdEdxMeanRatioPass3Neg,"Neg/Deft","P");
  legend->Draw();

  legend = new TLegend(0.595,0.25,0.75,0.40);
  legend->SetBorderSize(0);
  legend->SetFillColor(0);
  legend->SetTextFont(42);
  legend->SetTextSize(0.056);
  legend->AddEntry(hdEdxMeanRatioPass2,"Pos/NewFit","P");
  legend->AddEntry(hdEdxMeanRatioPass2Neg,"Neg/NewFit","P");
  legend->Draw();
  
  ///--------- Marking in momentum --------
  Float_t xBGLow = 0.2/fMass;
  Float_t xBGHigh= 7.0/fMass;
  Float_t yLinePos= 1.12; //10*funcPass2->Eval(1.0/fMass);


  ///Momentum Line, and Start and End ticks:
  drawMyline(xBGLow,yLinePos,xBGHigh,yLinePos, 1, 2, 1);
  drawMyline(xBGLow,yLinePos,xBGLow,yLinePos-0.02, 1, 2, 1);
  drawMyline(xBGHigh,yLinePos,xBGHigh,yLinePos-0.02, 1, 2, 1);

  
  tex = new TLatex(xBGLow*0.9,yLinePos+0.005,"0.2");
  tex->SetTextFont(42);
  tex->SetTextSize(0.06);
  tex->SetLineWidth(2);
  tex->Draw();

  tex = new TLatex(xBGLow*1.1,yLinePos-0.045,"p (GeV/c)");
  tex->SetTextFont(42);
  tex->SetTextSize(0.06);
  tex->SetLineWidth(2);
  tex->Draw();
  
  tex = new TLatex(xBGHigh*0.95,yLinePos+0.005,"7");
  tex->SetTextFont(42);
  tex->SetTextSize(0.06);
  tex->SetLineWidth(2);
  tex->Draw();
  
  //------Now Middle ticks:   
  xBGHigh= 0.5/fMass;
  drawMyline(xBGHigh,yLinePos,xBGHigh,yLinePos-0.02, 1, 2, 1);  
  tex = new TLatex(xBGHigh*0.95,yLinePos+0.005,"0.5");
  tex->SetTextFont(42);
  tex->SetTextSize(0.06);
  tex->SetLineWidth(2);
  tex->Draw();

  xBGHigh= 1.0/fMass;
  drawMyline(xBGHigh,yLinePos,xBGHigh,yLinePos-0.02, 1, 2, 1);  
  tex = new TLatex(xBGHigh*0.95,yLinePos+0.005,"1");
  tex->SetTextFont(42);
  tex->SetTextSize(0.06);
  tex->SetLineWidth(2);
  tex->Draw();

  xBGHigh= 2.0/fMass;
  drawMyline(xBGHigh,yLinePos,xBGHigh,yLinePos-0.02, 1, 2, 1);  
  tex = new TLatex(xBGHigh*0.95,yLinePos+0.005,"2");
  tex->SetTextFont(42);
  tex->SetTextSize(0.06);
  tex->SetLineWidth(2);
  tex->Draw();
  
  xBGHigh= 4.0/fMass;
  drawMyline(xBGHigh,yLinePos,xBGHigh,yLinePos-0.02, 1, 2, 1);  
  tex = new TLatex(xBGHigh*0.95,yLinePos+0.005,"4");
  tex->SetTextFont(42);
  tex->SetTextSize(0.06);
  tex->SetLineWidth(2);
  tex->Draw();
  //---------------------------
  
  
  
  if(Print)
    {
      std::string qaPath = CONFIG["output"]["fitBBGraph"]["QApath"].get<std::string>();
      TString qaPathT = TString::Format("%s", qaPath.c_str());
      if (!qaPathT.EndsWith("/")) qaPathT += "/";
      Canvas->SaveAs(Form("%sdEdxvsBetaGamma%s%s.pdf", qaPathT.Data(), name.Data(), passName.Data()));
    }
}










Double_t GetScoreRatio(TH1* hInput){
  Int_t NbinX = hInput->GetNbinsX();
  Double_t binCent,binCont, binErr, binWidth;
  Double_t sumScore=0;
    
  for(int i=1; i<=NbinX; i++){
    binCont = hInput->GetBinContent(i);
    if(binCont==0) continue;    
    binCent = hInput->GetBinCenter(i);
    binErr  = hInput->GetBinError(i);
    if(binErr>0.25) continue;
    sumScore += (1.0-binCont)*(1.0-binCont);
    //cout<<"indx:"<<i<<" binCent: "<<binCent<<"\t binErr: "<<binErr<<endl;    
  }
  if(sumScore>0)
    return sqrt(sumScore);
  else
    return -111;
}
  



