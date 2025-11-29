import ROOT
import os, sys, json
import array
import argparse

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
        raise FileNotFoundError(f"[ERROR]: ROOT file not found at path: {root_file_path}")
    else:
        LOG.info(f"Found Root tree at {root_file_path}")

    f = ROOT.TFile.Open(root_file_path, "READ")
    if not f or f.IsZombie():
        LOG.fatal(f"Could not open file: {root_file_path}")

    # Recursively collect all latest-cycle trees
    latest_trees = collect_latest_trees(f)
    trees = []
    for full_name, (_, tree) in latest_trees.items():
        # LOG.debug(f"Appending tree '{full_name}' of type: {type(tree)}")
        trees.append((full_name, tree))
        # LOG(f"Appended tree {full_name}")

    #DEBUG
    # for name, tree in trees:
    #     LOG.debug(f"Exporting tree '{name}' of type: {type(tree)}")

    return trees, f

#Method to verify that everything is good, and also prints leafs and branches of the tree
def check_trees(trees):

    for name, tree in trees:
        LOG.debug(f"CHECK tree '{name}' of type: {type(tree)}")
    """
    Diagnose and verify the list of (name, tree) tuples.
    Checks for None objects, wrong types, and prints structure.
    """
    if not trees:
        LOG.warning("Tree list is empty.")
        return

    LOG.info(f"Checking {len(trees)} trees...")

    for name, tree in trees:
        LOG.info(f"Tree: {name}")

        if tree is None:
            LOG.error(f"Tree object for '{name}' is None.")
            continue

        if not isinstance(tree, ROOT.TTree):
            LOG.error(f"Object for '{name}' is not a TTree (type: {type(tree).__name__})")
            continue

        LOG.info(f"Valid TTree with {tree.GetEntries()} entries.")

        # Try to list branches
        try:
            branches = tree.GetListOfBranches()
            if not branches:
                LOG.warning("No branches found.")
                continue

            branch_names = [branch.GetName() for branch in branches]
            LOG.info(f"Branches ({len(branch_names)}):")
            for branch in branches:
                LOG.info(f"    - {branch.GetName()}")


        except Exception as e:
            LOG.error(f"Exception while inspecting tree '{name}': {e}")

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

def count_particles_in_tree(tree, tree_name):
    """Print a per-species summary of entries stored in the provided tree."""
    if tree is None:
        LOG.warning(f"Tree {tree_name} is None. Skipping particle count.")
        return

    pid_leaf = tree.GetLeaf("fPidIndex")
    if not pid_leaf:
        LOG.warning(f"Tree {tree_name} has no fPidIndex leaf. Skipping particle count.")
        return

    counts = {
        "electrons": 0,
        "pions": 0,
        "kaons": 0,
        "protons": 0,
        "rest": 0,
    }

    nentries = tree.GetEntries()
    for i in range(nentries):
        tree.GetEntry(i)
        pid = int(pid_leaf.GetValue())
        if pid == 0:
            counts["electrons"] += 1
        elif pid == 2:
            counts["pions"] += 1
        elif pid == 3:
            counts["kaons"] += 1
        elif pid == 4:
            counts["protons"] += 1
        else:
            counts["rest"] += 1

    LOG.info(
        f"Tree {tree_name} contains: " +
        f"electrons={counts['electrons']}, pions={counts['pions']}, " +
        f"kaons={counts['kaons']}, protons={counts['protons']}, rest={counts['rest']}"
    )

def create_output_tree(tree_name, title, output_file):
    """Create an output tree with the standard branch layout used by the script."""
    output_file.cd()

    float_branch_names = [
        "fY",
        "fEta",
        "fPhi",
        "fTgl",
        "fMass",
        "fNSigTPC",
        "fNSigTOF",
        "fTPCSignal",
        "fSigned1Pt",
        "fBetaGamma",
        "fNormMultTPC",
        "fInvDeDxExpTPC",
        "fTPCInnerParam",
        "fNormNClustersTPC",
        "fFt0Occ",
        "fHadronicRate",
    ]
    int_branch_names = ["fPidIndex", "fRunNumber"]

    buffers = {}
    tree = ROOT.TTree(tree_name, title)

    for name in float_branch_names:
        buffers[name] = array.array('f', [0.0])
        tree.Branch(name, buffers[name], f"{name}/F")

    for name in int_branch_names:
        buffers[name] = array.array('i', [0])
        tree.Branch(name, buffers[name], f"{name}/I")

    return tree, buffers

def update_v0_tree(tree, name, calculate_dEdx, output_V0_tree, buffers):
    """
    Takes a V0 tree, updates the values, and fills the provided output tree.
    calculate_dEdx: callable func(beta_gamma) -> expected dEdx
    """
    if output_V0_tree is None or buffers is None:
        raise ValueError("Output tree and buffers must be provided for V0 update.")
    # LOG(f"Using dEdx values from branch {CONFIG['dataset']['dEdxSelection']} for V0 tree")
    
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
    fHadronicRate = array.array('f', [0.0])

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
    tree.SetBranchAddress("fHadronicRate", fHadronicRate)

    fsY = buffers["fY"]
    fsEta = buffers["fEta"]
    fsPhi = buffers["fPhi"]
    fsTgl = buffers["fTgl"]
    fsMass = buffers["fMass"]
    fsNSigTPC = buffers["fNSigTPC"]
    fsNSigTOF = buffers["fNSigTOF"]
    fsPidIndex = buffers["fPidIndex"]
    fsTPCSignal = buffers["fTPCSignal"]
    fsSigned1Pt = buffers["fSigned1Pt"]
    fsBetaGamma = buffers["fBetaGamma"]
    fsRunNumber = buffers["fRunNumber"]
    fsNormMultTPC = buffers["fNormMultTPC"]
    fsInvDeDxExpTPC = buffers["fInvDeDxExpTPC"]
    fsTPCInnerParam = buffers["fTPCInnerParam"]
    fsNormNClustersTPC = buffers["fNormNClustersTPC"]
    fsFt0Occ = buffers["fFt0Occ"]
    fsHadronicRate = buffers["fHadronicRate"]

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
        fsHadronicRate[0] = fHadronicRate[0]
        
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
        output_V0_tree.Fill()
    LOG.info("Particles found in V0 tree {name}: Electrons={nElectronV0}, Pions={nPionV0}, Kaons={nKaonV0}, Protons={nProtonV0}, Rest={nRestV0}")
    return True

def update_tpctof_tree(tree, name, calculate_dEdx, output_tpctof_tree, buffers):
    """
    Takes a TPCTOF tree, updates the values, and fills the provided output tree.
    calculate_dEdx: callable func(beta_gamma) -> expected dEdx
    """
    if output_tpctof_tree is None or buffers is None:
        raise ValueError("Output tree and buffers must be provided for TPCTOF update.")
    # LOG(f"Using dEdx values from branch {CONFIG['dataset']['dEdxSelection']} for tpctof tree")

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
    fHadronicRate = array.array('f', [0.0])

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
    tree.SetBranchAddress("fHadronicRate", fHadronicRate)

    fsY = buffers["fY"]
    fsEta = buffers["fEta"]
    fsPhi = buffers["fPhi"]
    fsTgl = buffers["fTgl"]
    fsMass = buffers["fMass"]
    fsNSigTPC = buffers["fNSigTPC"]
    fsNSigTOF = buffers["fNSigTOF"]
    fsPidIndex = buffers["fPidIndex"]
    fsTPCSignal = buffers["fTPCSignal"]
    fsSigned1Pt = buffers["fSigned1Pt"]
    fsBetaGamma = buffers["fBetaGamma"]
    fsRunNumber = buffers["fRunNumber"]
    fsNormMultTPC = buffers["fNormMultTPC"]
    fsInvDeDxExpTPC = buffers["fInvDeDxExpTPC"]
    fsTPCInnerParam = buffers["fTPCInnerParam"]
    fsNormNClustersTPC = buffers["fNormNClustersTPC"]
    fsFt0Occ = buffers["fFt0Occ"]
    fsHadronicRate = buffers["fHadronicRate"]

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
            # LOG.debug(f"Proton TPC Inner Param = {fTPCInnerParam[0]}")  
            # LOG.debug(f"Proton TPC abs(fSigned1Pt[0]) = {abs(1/fSigned1Pt[0])}")  
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
        fsHadronicRate[0] = fHadronicRate[0]

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


        output_tpctof_tree.Fill()

    LOG.info("Particles found in TPCTOF tree {name}: Electrons={nElectronV0}, Pions={nPionV0}, Kaons={nKaonV0}, Protons={nProtonV0}, Rest={nRestV0}")
    LOG.info("Rejected particles in TPCTOF tree {name}: RejectedPions={nRejectedPionsV0}, RejectedKaons={nRejectedKaonsV0}, RejectedProtons={nRejectedProtonsV0}")
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

    parser = argparse.ArgumentParser(description="Shift Nsigma values and apply cuts to ROOT trees.")
    parser.add_argument("--config", default="configuration.json", help="Path to the job directory containing config.txt")
    args = parser.parse_args()

    config = args.config
    with open(config, 'r') as config_file:
        CONFIG = json.load(config_file)
    sys.path.append(CONFIG['paths']['framework'] + "/framework")
    from base.config_tools import *

    LOG = logger.logger(min_severity=CONFIG["process"].get("severity", "DEBUG"), task_name="shift_nsigma_modified")

    #Reading 
    BB_params = CONFIG['output']['fitBBGraph']['BBparameters']

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
    write_config(CONFIG, path=args.config)

    v0_tree, v0_buffers = create_output_tree(CONFIG['general']['V0treename'], "Reconstructed ntuple", output_file)
    tpctof_tree, tpctof_buffers = create_output_tree(CONFIG['general']['tpctoftreename'], "Reconstructed ntuple", output_file)

    for name, tree in trees:
        # LOG(f"MAIN tree '{name}' of type: {type(tree)}")
        # count_particles_in_tree(tree, name)
        if name == CONFIG['general']['V0treename'] or name.endswith(f"/{CONFIG['general']['V0treename']}"):
            success_V0 = False
            success_V0 = update_v0_tree(tree, name, calculate_dEdx, v0_tree, v0_buffers)
            # LOG(f"Update of NSigma in V0 tree sucessful: {success_V0}")


        elif name == CONFIG['general']['tpctoftreename'] or name.endswith(f"/{CONFIG['general']['tpctoftreename']}"):
        # if name == "O2tpctofskimwde":
            success_tpctof = False
            success_tpctof = update_tpctof_tree(tree, name, calculate_dEdx, tpctof_tree, tpctof_buffers)
            # LOG(f"Update of NSigma in tpctof tree sucessful: {success_tpctof}")
        else:
            LOG.error(f"Unexpected tree {name}")
    count_particles_in_tree(v0_tree, "V0 output")
    count_particles_in_tree(tpctof_tree, "TPCTOF output")
    output_file.Write()
    output_file.Close()

    LOG.info(f"Skimmed Tree with updated NSigma and Cuts is stored in {output_file}")