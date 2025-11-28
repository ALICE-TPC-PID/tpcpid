import ROOT
import os, sys
import array
import argparse
import json

#This is the python equivalent to shiftNsigma.C
#This script takes as input the Data tree with the clean samples, and the BB parameters from the fitting macro.
#It then reads in the tree, and creates a copy
#Then it shifts the Nsigma values for the new tree, applies some cuts, creates some fits, and then stores the new tree

#one of the inputs is the path to the JOB, where the config.txt is stored
#for example /lustre/alice/users/jwitte/tpcpid/o2-tpcpid-parametrisation/BBfitAndQA/BBFitting_DEBUG/JOBS/20250523/32271368

#Reads the configurations from file
#Returns the config array
CONFIG = None   # module-level



def collect_latest_trees(directory, prefix=""):
    """Recursively collect the latest-cycle TTrees from a ROOT directory."""
    trees = {}
    for key in directory.GetListOfKeys():
        obj = key.ReadObj()
        name = obj.GetName()
        full_name = f"{prefix}/{name}" if prefix else name
        if obj.InheritsFrom("TDirectory"):
            # Recurse into subdirectory
            sub_trees = collect_latest_trees(obj, full_name)
            trees.update(sub_trees)
        elif obj.InheritsFrom("TTree"):
            cycle = key.GetCycle()
            # Keep only the latest cycle for each full_name
            if full_name not in trees or cycle > trees[full_name][0]:
                trees[full_name] = (cycle, obj)
    return trees


#Takes Config array
#Reads all trees, and automatically picks the newest onces
#Crashes if there is more than one subdirectory
#Returns array with tree names and the trees
def read_tree(config):
    root_file_path = config['dataset']['input_skimmedtree_path']
    if not root_file_path or not os.path.exists(root_file_path):
        raise FileNotFoundError(f"ROOT file not found at path: {root_file_path}")
    else:
        print(f"Found Root tree at {root_file_path}")

    f = ROOT.TFile.Open(root_file_path, "READ")
    if not f or f.IsZombie():
        print(f"Could not open file: {root_file_path}")
        exit(1)

    # Recursively collect all latest-cycle trees
    latest_trees = collect_latest_trees(f)
    trees = []
    for full_name, (_, tree) in latest_trees.items():
        # print(f"DEBUG: Appending tree '{full_name}' of type: {type(tree)}")
        trees.append((full_name, tree))

    #DEBUG
    # for name, tree in trees:
    #     print(f"DEBUG: Exporting tree '{name}' of type: {type(tree)}")

    return trees, f

#Method to verify that everything is good, and also prints leafs and branches of the tree
def check_trees(trees):

    for name, tree in trees:
        print(f"DEBUG: CHECK tree '{name}' of type: {type(tree)}")
    """
    Diagnose and verify the list of (name, tree) tuples.
    Checks for None objects, wrong types, and prints structure.
    """
    if not trees:
        print("[WARNING] Tree list is empty.")
        return

    print(f"[INFO] Checking {len(trees)} trees...")

    for name, tree in trees:
        print(f"\n[INFO] Tree: {name}")

        if tree is None:
            print(f"  [ERROR] Tree object for '{name}' is None.")
            continue

        if not isinstance(tree, ROOT.TTree):
            print(f"  [ERROR] Object for '{name}' is not a TTree (type: {type(tree).__name__})")
            continue

        print(f"  [OK] Valid TTree with {tree.GetEntries()} entries.")

        # Try to list branches
        try:
            branches = tree.GetListOfBranches()
            if not branches:
                print("  [WARNING] No branches found.")
                continue

            branch_names = [branch.GetName() for branch in branches]
            print(f"  [INFO] Branches ({len(branch_names)}):")
            for branch in branches:
                print(f"    - {branch.GetName()}")


        except Exception as e:
            print(f"  [ERROR] Exception while inspecting tree '{name}': {e}")

def create_funcBBvsBGNew(BB_params):
    """
    Creates a TF1 function using the BB_params and returns a callable function
    that takes beta_gamma as input and returns the corresponding dEdx value.
    """
    # Create the TF1 function
    funcBBvsBGNew = ROOT.TF1(
        "funcBBvsBGNew",
        "[0]*([1]- log([2]+pow(x,-1.*[4])) - pow((x/sqrt(1+x*x)),[3]))*pow(1.0,2.3)*50./pow((x/sqrt(1+x*x)),[3])",
        0.5e-3,
        4.E5
    )

    # Set the parameters from BB_params
    for i, param in enumerate(BB_params):
        funcBBvsBGNew.SetParameter(i, param)

    # Return a callable function
    def calculate_dEdx(beta_gamma):
        return funcBBvsBGNew.Eval(beta_gamma)

    return calculate_dEdx


def update_v0_tree(tree, calculate_dEdx, output_file):
    """
    Takes a V0 tree, updates the values, and returns a new tree.
    calculate_dEdx: callable func(beta_gamma) -> expected dEdx
    """
    output_file.cd()
    print(f"Using dEdx values from branch {CONFIG['dataset']['dEdxSelection']} for V0 tree")

    # Prepare input buffers for SetBranchAddress
    fY = array.array('f', [0.0])
    fEta = array.array('f', [0.0])
    fPhi = array.array('f', [0.0])
    fTgl = array.array('f', [0.0])
    fMass = array.array('f', [0.0])
    fNSigTPC = array.array('f', [0.0])
    fNSigTOF = array.array('f', [0.0])
    fPidIndex = array.array('i', [0])
    fTPCSignal = array.array('f', [0.0])
    fTPCdEdxNorm = array.array('f', [0.0])
    fSigned1Pt = array.array('f', [0.0])
    fBetaGamma = array.array('f', [0.0])
    fRunNumber = array.array('i', [0])
    fNormMultTPC = array.array('f', [0.0])
    fInvDeDxExpTPC = array.array('f', [0.0])
    fTPCInnerParam = array.array('f', [0.0])
    fNormNClustersTPC = array.array('f', [0.0])
    fFt0Occ = array.array('f', [0.0])

    # Connect input branches
    tree.SetBranchAddress("fY", fY)
    tree.SetBranchAddress("fEta", fEta)
    tree.SetBranchAddress("fPhi", fPhi)
    tree.SetBranchAddress("fTgl", fTgl)
    tree.SetBranchAddress("fMass", fMass)
    tree.SetBranchAddress("fNSigTPC", fNSigTPC)
    tree.SetBranchAddress("fNSigTOF", fNSigTOF)
    tree.SetBranchAddress("fPidIndex", fPidIndex)
    tree.SetBranchAddress("fTPCSignal", fTPCSignal)
    tree.SetBranchAddress("fTPCdEdxNorm", fTPCdEdxNorm)
    tree.SetBranchAddress("fSigned1Pt", fSigned1Pt)
    tree.SetBranchAddress("fBetaGamma", fBetaGamma)
    tree.SetBranchAddress("fRunNumber", fRunNumber)
    tree.SetBranchAddress("fNormMultTPC", fNormMultTPC)
    tree.SetBranchAddress("fInvDeDxExpTPC", fInvDeDxExpTPC)
    tree.SetBranchAddress("fTPCInnerParam", fTPCInnerParam)
    tree.SetBranchAddress("fNormNClustersTPC", fNormNClustersTPC)
    tree.SetBranchAddress("fFt0Occ", fFt0Occ)

    # Prepare output arrays (same types)
    fsY = array.array('f', [0.0])
    fsEta = array.array('f', [0.0])
    fsPhi = array.array('f', [0.0])
    fsTgl = array.array('f', [0.0])
    fsMass = array.array('f', [0.0])
    fsNSigTPC = array.array('f', [0.0])
    fsNSigTOF = array.array('f', [0.0])
    fsPidIndex = array.array('i', [0])
    fsTPCSignal = array.array('f', [0.0])
    fsSigned1Pt = array.array('f', [0.0])
    fsBetaGamma = array.array('f', [0.0])
    fsRunNumber = array.array('i', [0])
    fsNormMultTPC = array.array('f', [0.0])
    fsInvDeDxExpTPC = array.array('f', [0.0])
    fsTPCInnerParam = array.array('f', [0.0])
    fsNormNClustersTPC = array.array('f', [0.0])
    fsFt0Occ = array.array('f', [0.0])

    # Create output tree and branches
    gTree_V0 = ROOT.TTree("O2V0Tree", "Reconstructed ntuple")
    gTree_V0.Branch("fY", fsY, "fY/F")
    gTree_V0.Branch("fEta", fsEta, "fEta/F")
    gTree_V0.Branch("fPhi", fsPhi, "fPhi/F")
    gTree_V0.Branch("fTgl", fsTgl, "fTgl/F")
    gTree_V0.Branch("fMass", fsMass, "fMass/F")
    gTree_V0.Branch("fNSigTPC", fsNSigTPC, "fNSigTPC/F")
    gTree_V0.Branch("fNSigTOF", fsNSigTOF, "fNSigTOF/F")
    gTree_V0.Branch("fPidIndex", fsPidIndex, "fPidIndex/I")
    gTree_V0.Branch("fTPCSignal", fsTPCSignal, "fTPCSignal/F")
    gTree_V0.Branch("fSigned1Pt", fsSigned1Pt, "fSigned1Pt/F")
    gTree_V0.Branch("fBetaGamma", fsBetaGamma, "fBetaGamma/F")
    gTree_V0.Branch("fRunNumber", fsRunNumber, "fRunNumber/I")
    gTree_V0.Branch("fNormMultTPC", fsNormMultTPC, "fNormMultTPC/F")
    gTree_V0.Branch("fInvDeDxExpTPC", fsInvDeDxExpTPC, "fInvDeDxExpTPC/F")
    gTree_V0.Branch("fTPCInnerParam", fsTPCInnerParam, "fTPCInnerParam/F")
    gTree_V0.Branch("fNormNClustersTPC", fsNormNClustersTPC, "fNormNClustersTPC/F")
    gTree_V0.Branch("fFt0Occ", fsFt0Occ, "fFt0Occ/F")

    nentries = tree.GetEntries()

    # for i in range(1,100):
    #     # print(f"fPID index for entry ")
    #     print(f"Event {i}")
    #     print(tree.Show(i))
        # tree.GetEntry(i)
        # print(tree.GetEntry(i))
        # print(f"fPID index for entry {i} = {tree.fPidIndex}")

    #add particle counter
    nElectronV0 = 0
    nPionV0 = 0
    nKaonV0 = 0
    nProtonV0 = 0
    nRestV0 = 0

    for i in range(nentries):
        tree.GetEntry(i)

        #count particles by species
        if fPidIndex[0] == 0:
            nElectronV0 += 1
        elif fPidIndex[0] == 2:
            nPionV0 += 1
        elif fPidIndex[0] == 3:
            nKaonV0 += 1
        elif fPidIndex[0] == 4:
            nProtonV0 += 1
        else:
            nRestV0 += 1

            # return False

        # Copy values to output arrays
        fsY[0] = fY[0]
        fsEta[0] = fEta[0]
        fsPhi[0] = fPhi[0]
        fsTgl[0] = fTgl[0]
        fsMass[0] = fMass[0]
        fsNSigTOF[0] = fNSigTOF[0]
        fsPidIndex[0] = fPidIndex[0]
        fsSigned1Pt[0] = fSigned1Pt[0]
        fsBetaGamma[0] = fBetaGamma[0]
        fsRunNumber[0] = fRunNumber[0]
        fsNormMultTPC[0] = fNormMultTPC[0]
        fsTPCInnerParam[0] = fTPCInnerParam[0]
        fsNormNClustersTPC[0] = fNormNClustersTPC[0]
        fsFt0Occ[0] = fFt0Occ[0]

        expected_dEdx = calculate_dEdx(fsBetaGamma[0])
        fsInvDeDxExpTPC[0] = 1.0 / expected_dEdx if expected_dEdx != 0 else 0

        if CONFIG['dataset']['dEdxSelection'] == "TPCSignal":
            fsTPCSignal[0] = fTPCSignal[0]
        elif CONFIG['dataset']['dEdxSelection'] == "TPCdEdxNorm":
            fsTPCSignal[0] = fTPCdEdxNorm[0]

        fsNSigTPC[0] = (fsTPCSignal[0] - expected_dEdx) / (0.07 * expected_dEdx) if expected_dEdx != 0 else 0

        # tree.Show(i)
        # print(f"PID value = {fsPidIndex[0]}")
        # print(f"TPCSignal = {fTPCSignal[0]}")
        # Calculate expected dEdx and update InvDeDxExpTPC and NSigTPC
        # print(f"expected_dEdx = {expected_dEdx}")

        #writes every single entry to the output root tree
        gTree_V0.Fill()
    print("[CRITICAL]: Just one tree is saved, they are overwritten")
    print(f"Particles found in V0 tree: Electrons={nElectronV0}, Pions={nPionV0}, Kaons={nKaonV0}, Protons={nProtonV0}, Rest={nRestV0}")
    gTree_V0.Write()
    return True

def update_tpctof_tree(tree, calculate_dEdx, output_file):
    """
    Takes a TPCTOF tree, updates the values, and returns a new tree.
    calculate_dEdx: callable func(beta_gamma) -> expected dEdx
    """
    output_file.cd()
    print(f"Using dEdx values from branch {CONFIG['dataset']['dEdxSelection']} for tpctof tree")

    # Prepare input buffers for SetBranchAddress
    fY = array.array('f', [0.0])
    fEta = array.array('f', [0.0])
    fPhi = array.array('f', [0.0])
    fTgl = array.array('f', [0.0])
    fMass = array.array('f', [0.0])
    fNSigTPC = array.array('f', [0.0])
    fNSigTOF = array.array('f', [0.0])
    fPidIndex = array.array('i', [0])
    fTPCSignal = array.array('f', [0.0])
    fTPCdEdxNorm = array.array('f', [0.0])
    fSigned1Pt = array.array('f', [0.0])
    fBetaGamma = array.array('f', [0.0])
    fRunNumber = array.array('i', [0])
    fNormMultTPC = array.array('f', [0.0])
    fInvDeDxExpTPC = array.array('f', [0.0])
    fTPCInnerParam = array.array('f', [0.0])
    fNormNClustersTPC = array.array('f', [0.0])
    fFt0Occ = array.array('f', [0.0])

    # Connect input branches
    tree.SetBranchAddress("fY", fY)
    tree.SetBranchAddress("fEta", fEta)
    tree.SetBranchAddress("fPhi", fPhi)
    tree.SetBranchAddress("fTgl", fTgl)
    tree.SetBranchAddress("fMass", fMass)
    tree.SetBranchAddress("fNSigTPC", fNSigTPC)
    tree.SetBranchAddress("fNSigTOF", fNSigTOF)
    tree.SetBranchAddress("fPidIndex", fPidIndex)
    tree.SetBranchAddress("fTPCSignal", fTPCSignal)
    tree.SetBranchAddress("fTPCdEdxNorm", fTPCdEdxNorm)
    tree.SetBranchAddress("fSigned1Pt", fSigned1Pt)
    tree.SetBranchAddress("fBetaGamma", fBetaGamma)
    tree.SetBranchAddress("fRunNumber", fRunNumber)
    tree.SetBranchAddress("fNormMultTPC", fNormMultTPC)
    tree.SetBranchAddress("fInvDeDxExpTPC", fInvDeDxExpTPC)
    tree.SetBranchAddress("fTPCInnerParam", fTPCInnerParam)
    tree.SetBranchAddress("fNormNClustersTPC", fNormNClustersTPC)
    tree.SetBranchAddress("fFt0Occ", fFt0Occ)

    # Prepare output arrays (same types)
    fsY = array.array('f', [0.0])
    fsEta = array.array('f', [0.0])
    fsPhi = array.array('f', [0.0])
    fsTgl = array.array('f', [0.0])
    fsMass = array.array('f', [0.0])
    fsNSigTPC = array.array('f', [0.0])
    fsNSigTOF = array.array('f', [0.0])
    fsPidIndex = array.array('i', [0])
    fsTPCSignal = array.array('f', [0.0])
    fsSigned1Pt = array.array('f', [0.0])
    fsBetaGamma = array.array('f', [0.0])
    fsRunNumber = array.array('i', [0])
    fsNormMultTPC = array.array('f', [0.0])
    fsInvDeDxExpTPC = array.array('f', [0.0])
    fsTPCInnerParam = array.array('f', [0.0])
    fsNormNClustersTPC = array.array('f', [0.0])
    fsFt0Occ = array.array('f', [0.0])

    # Create output tree and branches
    gTree_tpctof = ROOT.TTree("O2tpctofTree", "Reconstructed ntuple")
    gTree_tpctof.Branch("fY", fsY, "fY/F")
    gTree_tpctof.Branch("fEta", fsEta, "fEta/F")
    gTree_tpctof.Branch("fPhi", fsPhi, "fPhi/F")
    gTree_tpctof.Branch("fTgl", fsTgl, "fTgl/F")
    gTree_tpctof.Branch("fMass", fsMass, "fMass/F")
    gTree_tpctof.Branch("fNSigTPC", fsNSigTPC, "fNSigTPC/F")
    gTree_tpctof.Branch("fNSigTOF", fsNSigTOF, "fNSigTOF/F")
    gTree_tpctof.Branch("fPidIndex", fsPidIndex, "fPidIndex/I")
    gTree_tpctof.Branch("fTPCSignal", fsTPCSignal, "fTPCSignal/F")
    gTree_tpctof.Branch("fSigned1Pt", fsSigned1Pt, "fSigned1Pt/F")
    gTree_tpctof.Branch("fBetaGamma", fsBetaGamma, "fBetaGamma/F")
    gTree_tpctof.Branch("fRunNumber", fsRunNumber, "fRunNumber/I")
    gTree_tpctof.Branch("fNormMultTPC", fsNormMultTPC, "fNormMultTPC/F")
    gTree_tpctof.Branch("fInvDeDxExpTPC", fsInvDeDxExpTPC, "fInvDeDxExpTPC/F")
    gTree_tpctof.Branch("fTPCInnerParam", fsTPCInnerParam, "fTPCInnerParam/F")
    gTree_tpctof.Branch("fNormNClustersTPC", fsNormNClustersTPC, "fNormNClustersTPC/F")
    gTree_tpctof.Branch("fFt0Occ", fsFt0Occ, "fFt0Occ/F")

    nentries = tree.GetEntries()

    #add particle counter
    nElectronV0 = 0
    nPionV0 = 0
    nKaonV0 = 0
    nProtonV0 = 0
    nRestV0 = 0
    nRejectedPionsV0= 0
    nRejectedKaonsV0= 0
    nRejectedProtonsV0= 0

    for i in range(nentries):
        tree.GetEntry(i)
        #count particles by species
        if fPidIndex[0] == 0:
            nElectronV0 += 1
        elif fPidIndex[0] == 2:
            nPionV0 += 1
        elif fPidIndex[0] == 3:
            nKaonV0 += 1
        elif fPidIndex[0] == 4:
            nProtonV0 += 1
            # print(f"DEBUG: Proton TPC Inner Param = {fTPCInnerParam[0]}")
            # print(f"DEBUG: Proton TPC abs(fSigned1Pt[0]) = {abs(1/fSigned1Pt[0])}")
        else:
            nRestV0 += 1

        # Copy values to output arrays
        fsY[0] = fY[0]
        fsEta[0] = fEta[0]
        fsPhi[0] = fPhi[0]
        fsTgl[0] = fTgl[0]
        fsMass[0] = fMass[0]
        fsNSigTOF[0] = fNSigTOF[0]
        fsPidIndex[0] = fPidIndex[0]
        fsSigned1Pt[0] = fSigned1Pt[0]
        fsBetaGamma[0] = fBetaGamma[0]
        fsRunNumber[0] = fRunNumber[0]
        fsNormMultTPC[0] = fNormMultTPC[0]
        fsTPCInnerParam[0] = fTPCInnerParam[0]
        fsNormNClustersTPC[0] = fNormNClustersTPC[0]
        fsFt0Occ[0] = fFt0Occ[0]

        expected_dEdx = calculate_dEdx(fsBetaGamma[0])
        fsInvDeDxExpTPC[0] = 1.0 / expected_dEdx if expected_dEdx != 0 else 0

        if CONFIG['dataset']['dEdxSelection'] == "TPCSignal":
            fsTPCSignal[0] = fTPCSignal[0]
        elif CONFIG['dataset']['dEdxSelection'] == "TPCdEdxNorm":
            fsTPCSignal[0] = fTPCdEdxNorm[0]

        fsNSigTPC[0] = (fsTPCSignal[0] - expected_dEdx) / (0.07 * expected_dEdx) if expected_dEdx != 0 else 0

        # print(f"Original TPCNSigma {fNSigTPC}")
        # print(f"New TPCNSigma {fsNSigTPC}")

        #Here is the section where we apply cuts, on the kinematics and the PID
        if(fPidIndex[0]==2 and fTPCInnerParam[0] > 1.8):
            # print(f"Rejected TPC inner parameter is {fTPCInnerParam[0]} ")
            nRejectedPionsV0 += 1
            continue
        if(fPidIndex[0]==4 and fTPCInnerParam[0] > 0.8):
            nRejectedProtonsV0 += 1
            # print("[DEBUG] Cut on Inner Param applied")
            continue
        if(fPidIndex[0]==2 and (abs(1/fSigned1Pt[0]))> 1.8):
            nRejectedPionsV0 += 1
            continue
        if(fPidIndex[0]==4 and (abs(1/fSigned1Pt[0]))> 0.8):
            nRejectedProtonsV0 += 1
            # print("[DEBUG] Cut on Signed1Pt applied")
            continue

        #New Kaon cuts, to look at distribution and contamination
        if(fPidIndex[0]==3 and (fsNSigTPC[0]>2.5)):
            nRejectedKaonsV0 += 1
            continue
        if(fPidIndex[0]==3 and (fsNSigTPC[0]<-2.5)):
            nRejectedKaonsV0 += 1
            continue


        gTree_tpctof.Fill()

    print(f"Particles found in TPCTOF tree: Electrons={nElectronV0}, Pions={nPionV0}, Kaons={nKaonV0}, Protons={nProtonV0}, Rest={nRestV0}")
    print(f"Rejected particles in TPCTOF tree: RejectedPions={nRejectedPionsV0}, RejectedKaons={nRejectedKaonsV0}, RejectedProtons={nRejectedProtonsV0}")
    gTree_tpctof.Write()
    return True




def collect_trees(root_dir, prefix=""):
    """Recursively collect all trees in a ROOT directory."""
    trees = []
    for key in root_dir.GetListOfKeys():
        obj = key.ReadObj()
        name = obj.GetName()
        full_name = f"{prefix}/{name}" if prefix else name
        if obj.InheritsFrom("TDirectory"):
            trees.extend(collect_trees(obj, full_name))
        elif obj.InheritsFrom("TTree"):
            trees.append((full_name, obj))
    return trees


if __name__ == "__main__":
    # Define the job directory
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Shift Nsigma values and apply cuts to ROOT trees.")
    parser.add_argument("--config", default="configuration.json", help="Path to the job directory containing config.txt")

    # Parse arguments
    args = parser.parse_args()
    config = args.config
    with open(config, 'r') as config_file:
        CONFIG = json.load(config_file)

    sys.path.append(os.path.join(CONFIG["output"]["general"]["base_folder"], "framework"))
    from base.config_tools import *

    # Job_dir = "/lustre/alice/users/jwitte/tpcpid/o2-tpcpid-parametrisation/BBfitAndQA/BBFitting_DEBUG/JOBS/20250528/32632827"
    Job_dir = CONFIG["output"]["general"]["path"]
    print(f"Job dir = {Job_dir}")

    # Read the configuration
    CONFIG["dataset"]["Job_dir"] = Job_dir

    # #define to use tpcsignal or tpcdEdxNorm
    # CONFIG['dataset']['dEdxSelection'] = "TPCSignal"
    # CONFIG['dataset']['dEdxSelection'] = "TPCdEdxNorm"
    # print(f"Using Hardcoded dEdx values in shiftNSigma {config['dEdxSelection']}")

    #Reading
    BB_params = CONFIG['output']['fitBBGraph']['BBparameters']
    print(f"BB parameters used for dEdx calculation: {BB_params}")

    #create function to calculate dEdx values from fit
    calculate_dEdx = create_funcBBvsBGNew(BB_params)




    # Read the tree from the ROOT file
    #f is the loaded tree, that needs to be kept alive
    trees, f = read_tree(CONFIG)           #works, 26.05.25

    # #Check content of trees
    # check_trees(trees)

    output_file = ROOT.TFile(os.path.join(CONFIG['output']['general']['path'],"trees",f"SkimmedTree_UpdatednSigmaAndExpdEdx_{CONFIG['output']['general']['name']}.root"), "RECREATE")
    CONFIG["output"].setdefault('shiftNsigma', {})
    CONFIG["output"]["shiftNsigma"]["Skimmedtree_shiftedNsigma_path"] = os.path.join(CONFIG['output']['general']['path'],"trees",f"SkimmedTree_UpdatednSigmaAndExpdEdx_{CONFIG['output']['general']['name']}.root")
    CONFIG["output"]["shiftNsigma"]["training_data"] = os.path.join(CONFIG['output']['general']['path'],"trees/merged_tree_for_training.root")
    write_config(CONFIG, path=args.config)

    for name, tree in trees:
        # print(f"DEBUG: MAIN tree '{name}' of type: {type(tree)}")
        if name == CONFIG['general']['V0treename'] or name.endswith(f"/{CONFIG['general']['V0treename']}"):
            success_V0 = False
            success_V0 = update_v0_tree(tree, calculate_dEdx, output_file)
            print(f"Update of NSigma in V0 tree sucessful: {success_V0}")


        elif name == CONFIG['general']['tpctoftreename'] or name.endswith(f"/{CONFIG['general']['tpctoftreename']}"):
        # if name == "O2tpctofskimwde":
            success_tpctof = False
            success_tpctof = update_tpctof_tree(tree, calculate_dEdx, output_file)
            print(f"Update of NSigma in tpctof tree sucessful: {success_tpctof}")
        else:
            print(f"Unexpected tree {name}")

    output_file.Write()
    output_file.Close()

    print(f"Skimmed Tree with updated NSigma and Cuts is stored in {output_file}")