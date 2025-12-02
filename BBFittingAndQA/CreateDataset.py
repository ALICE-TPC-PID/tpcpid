import os
import sys
from datetime import datetime
import json
import numpy as np
import scipy as sc
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import tqdm
import matplotlib.colors as mcolors
import pandas as pd
import argparse
import mplhep as hep
import pathlib
import sys

from config_tools import (
    add_name_and_path,
    read_config,
    write_config,
)




plt.style.use(hep.style.ALICE)
for key in mpl.rcParams.keys():
    if key.startswith('legend.'):
        mpl.rcParams[key] = mpl.rcParamsDefault[key]

print("--- Starting the data preparation script ---\n")
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output-path", default="training_data", help="Output absolute path for the merged tree")
parser.add_argument("-s", "--output-size", default="10000000", help="Output size for the merged tree.")
parser.add_argument("-period", "--period", default=";;", help="Period to be processed, please check the script and adjust dir_tree if necessary!")
parser.add_argument("-apass", "--apass", default=";;", help="Period to be processed, please check the script and adjust dir_tree if necessary!")
parser.add_argument("-lm", "--loading-mode", default="full", help="Loading mode for the trees: 'full' -> Load all trees into one variable, 'separate' -> Loads V0 and TPC-TOF trees separately and applies V0 selections to V0 tree")
parser.add_argument("-f", "--full-input-path", default=";;", help="If --full-path != ';;', then ignore -apass and -period and use the full path (this variable) instead.")
parser.add_argument("-ce", "--cut-electrons", default="[np.log10(0.11),np.log10(5.)]", help="Momentum range for electrons.")
parser.add_argument("-cpi", "--cut-pions", default="[np.log10(0.11),np.log10(20.)]", help="Momentum range for pions.")
parser.add_argument("-cka", "--cut-kaons", default="[np.log10(0.12),np.log10(2.)]", help="Momentum range for kaons.")
parser.add_argument("-cp", "--cut-protons", default="[np.log10(0.12),np.log10(15.)]", help="Momentum range for protons.")
parser.add_argument("-cd", "--cut-deuterons", default="[np.log10(0.3),np.log10(2.)]", help="Momentum range for deuterons.")
parser.add_argument("-ct", "--cut-tritons", default="[np.log10(0.3),np.log10(1.)]", help="Momentum range for tritons.")
parser.add_argument("-sg","--sigmathreshold", type= int, default=3, help= "Define the sigmathreshold of what data should be selected")
parser.add_argument("-sa","--sampleamount", type= int, default=6e7, help= "Define the max amount of events before downsampling")
args = parser.parse_args()

# if args.full_input_path == ";;" and (args.period == ";;" or args.apass == ";;"):
#     print("Please provide either a full path or period and apass.")
#     sys.exit()

# if args.full_input_path == ";;":
#     period = args.period
#     apass = args.apass
#     dir_tree = "/lustre/alice/users/msalvan/o2-tpcpid-parametrisation/BBfitAndQA/{0}_{1}/outputFits/SkimmedTree_UpdatednSigmaAndExpdEdx_Single_{0}_{1}.root".format(period, apass)
# else:
#     period = "default"
#     apass = "default"
#     dir_tree = args.full_input_path
#################################

CONFIG = read_config()

neuralNetClass_dir = os.path.join(CONFIG['output']['general']['base_folder'],"Neural-Network-Class","NeuralNetworkClasses")
sys.path.append(neuralNetClass_dir)
# print(f"files in folder neuralNetClass_dir = {os.listdir(neuralNetClass_dir)}")
from extract_from_root import *


period = CONFIG['dataset']['period']
apass = CONFIG['dataset']['pass']

date = datetime.today().strftime('%d%m%Y')
output_path = os.path.join(CONFIG['output']['general']['path'],"trees","merged_tree_for_training.root")
CONFIG["output"]["createTrainingDataset"]["training_data"] = output_path
write_config(CONFIG)

plot_path = os.path.join(CONFIG['output']['general']['path'], "QA", "createTrainingDataset")

dir_tree = CONFIG['output']['shiftNsigma']['Skimmedtree_shiftedNsigma_path']
print("Period:", period, "; apass:", apass, "; input is:", dir_tree)


### Functions

def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

### V0 cleaning

cut_dict = {
    # Gamma cuts
    "cutAlphaG": 0.4,
    "cutQTG": 0.006,
    "cutAlphaGLow": 0.4,
    "cutAlphaGHigh": 0.8,
    "cutQTG2": 0.006,

    # K0S cuts
    "cutQTK0SLow": 0.1075,
    "cutQTK0SHigh": 0.215,
    "cutAPK0SLow": 0.199,
    "cutAPK0SHigh": 0.8,
    "cutAPK0SHighTop": 1.,

    # Lambda & Anti-Lambda cuts
    "cutQTL": 0.03,
    "cutAlphaLLow": 0.35,
    "cutAlphaLLow2": 0.53,
    "cutAlphaLHigh": 0.7,
    "cutAPL1": 0.107,
    "cutAPL2": -0.69,
    "cutAPL3": 0.5,
    "cutAPL1Low": 0.091,
    "cutAPL2Low": -0.69,
    "cutAPL3Low": 0.156
}

particle_type = {
    "kGamma": 1,
    "kK0S": 2,
    "kLambda": 3,
    "kAntiLambda": 4,
    "kUndef": 0
}

def checkV0(alpha, qt, **kwargs):

    cutAlphaG = kwargs["cutAlphaG"]
    cutQTG = kwargs["cutQTG"]
    cutAlphaGLow = kwargs["cutAlphaGLow"]
    cutAlphaGHigh = kwargs["cutAlphaGHigh"]
    cutQTG2 = kwargs["cutQTG2"]
    cutQTK0SLow = kwargs["cutQTK0SLow"]
    cutQTK0SHigh = kwargs["cutQTK0SHigh"]
    cutAPK0SLow = kwargs["cutAPK0SLow"]
    cutAPK0SHigh = kwargs["cutAPK0SHigh"]
    cutAPK0SHighTop = kwargs["cutAPK0SHighTop"]
    cutQTL = kwargs["cutQTL"]
    cutAlphaLLow = kwargs["cutAlphaLLow"]
    cutAlphaLLow2 = kwargs["cutAlphaLLow2"]
    cutAlphaLHigh = kwargs["cutAlphaLHigh"]
    cutAPL1 = kwargs["cutAPL1"]
    cutAPL2 = kwargs["cutAPL2"]
    cutAPL3 = kwargs["cutAPL3"]
    cutAPL1Low = kwargs["cutAPL1Low"]
    cutAPL2Low = kwargs["cutAPL2Low"]
    cutAPL3Low = kwargs["cutAPL3Low"]

    GAMMAS = ((qt < cutQTG)*(np.abs(alpha) < cutAlphaG)) + ((qt < cutQTG2) * (cutAlphaGLow < np.abs(alpha)) * (np.abs(alpha) < cutAlphaGHigh))

    # Check for K0S candidates
    qtop =  cutQTK0SHigh * np.sqrt(np.abs(1. - alpha * alpha / (cutAPK0SHighTop * cutAPK0SHighTop)))
    q = cutAPK0SLow * np.sqrt(np.abs(1 - alpha**2 / (cutAPK0SHigh**2)))
    K0S = (cutQTK0SLow < qt) * (qt < cutQTK0SHigh) * (qt < cutAPK0SHighTop)  * (qtop > qt) * (q < qt)

    # Check for Lambda candidates
    q = cutAPL1 * np.sqrt(np.abs(1 - ((alpha + cutAPL2)**2) / (cutAPL3**2))) * (cutAlphaLLow < alpha)
    q_2 = cutAPL1Low * np.sqrt(np.abs(1 - ((alpha + cutAPL2Low)**2) / (cutAPL3Low**2))) * (cutAlphaLLow2 < alpha)
    LAMBDAS = (alpha < cutAlphaLHigh) * (cutQTL < qt) * (q > qt) * (q_2 < qt)

    # Check for Anti-Lambda candidates
    q = cutAPL1 * np.sqrt(np.abs(1 - ((alpha - cutAPL2)**2) / (cutAPL3**2))) * (alpha < -cutAlphaLLow)
    q_2 = cutAPL1Low * np.sqrt(np.abs(1 - ((alpha - cutAPL2Low)**2 / (cutAPL3Low**2)))) * (alpha < -cutAlphaLLow2)
    ANTILAMBDAS = (-cutAlphaLHigh < alpha) * (cutQTL < qt) * (q > qt) * (q_2 < qt)

    return K0S, LAMBDAS, ANTILAMBDAS, GAMMAS

def plot_cuts(**kwargs):
    alpha = np.linspace(-1.05, 1.05, 1000)

    cutAlphaG = kwargs["cutAlphaG"]
    cutQTG = kwargs["cutQTG"]
    cutAlphaGLow = kwargs["cutAlphaGLow"]
    cutAlphaGHigh = kwargs["cutAlphaGHigh"]
    cutQTG2 = kwargs["cutQTG2"]
    cutQTK0SLow = kwargs["cutQTK0SLow"]
    cutQTK0SHigh = kwargs["cutQTK0SHigh"]
    cutAPK0SLow = kwargs["cutAPK0SLow"]
    cutAPK0SHigh = kwargs["cutAPK0SHigh"]
    cutAPK0SHighTop = kwargs["cutAPK0SHighTop"]
    cutQTL = kwargs["cutQTL"]
    cutAlphaLLow = kwargs["cutAlphaLLow"]
    cutAlphaLLow2 = kwargs["cutAlphaLLow2"]
    cutAlphaLHigh = kwargs["cutAlphaLHigh"]
    cutAPL1 = kwargs["cutAPL1"]
    cutAPL2 = kwargs["cutAPL2"]
    cutAPL3 = kwargs["cutAPL3"]
    cutAPL1Low = kwargs["cutAPL1Low"]
    cutAPL2Low = kwargs["cutAPL2Low"]
    cutAPL3Low = kwargs["cutAPL3Low"]

    # K0S cut
    def K0S_CUT(alpha):
        q = cutAPK0SLow * np.sqrt(np.abs(1 - alpha**2 / (cutAPK0SHigh**2)))
        q[~((cutQTK0SLow < q) * (q < cutQTK0SHigh))] = np.nan
        return q
    plt.plot(alpha, K0S_CUT(alpha), label="K0S Cut", color="black", linewidth = 4)

    def K0S_CUT_UPPER(alpha):
        q =  cutQTK0SHigh * np.sqrt(np.abs(1. - alpha**2 / (cutAPK0SHighTop**2)))
        q[~((cutQTK0SLow < q) * (q < cutQTK0SHigh) * (q < cutAPK0SHighTop))] = np.nan
        return q
    plt.plot(alpha, K0S_CUT_UPPER(alpha), label="K0S Cut", color="black", linewidth = 4)

    # Lambda cut
    def LAMBDA_CUT(alpha):
        q = cutAPL1 * np.sqrt(np.abs(1 - ((alpha + cutAPL2)**2) / (cutAPL3**2))) * (cutAlphaLLow < alpha)
        q[~((alpha < cutAlphaLHigh) * (cutQTL < q))] = np.nan
        return q
    plt.plot(alpha, LAMBDA_CUT(alpha), label="Lambda Cut", color="red", linewidth = 4)

    def LAMBDA_CUT_LOW(alpha):
        q = cutAPL1Low * np.sqrt(np.abs(1 - ((alpha + cutAPL2Low)**2) / (cutAPL3Low**2))) * (cutAlphaLLow2 < alpha)
        q[~((alpha < cutAlphaLHigh) * (cutQTL < q))] = np.nan
        return q
    plt.plot(alpha, LAMBDA_CUT_LOW(alpha), label="Lambda Cut", color="red", linewidth = 4)

    # Anti-Lambda cut
    def ANTILAMBDA_CUT(alpha):
        q = cutAPL1 * np.sqrt(np.abs(1 - ((alpha - cutAPL2)**2 / (cutAPL3**2)))) * (alpha < -cutAlphaLLow)
        q[~((-cutAlphaLHigh < alpha) * (cutQTL < q))] = np.nan
        return q
    plt.plot(alpha, ANTILAMBDA_CUT(alpha), label="Anti-Lambda Cut", color="red", linewidth = 4)

    def ANTILAMBDA_CUT_LOW(alpha):
        q = cutAPL1Low * np.sqrt(np.abs(1 - ((alpha - cutAPL2Low)**2 / (cutAPL3Low**2)))) * (alpha < -cutAlphaLLow2)
        q[~((-cutAlphaLHigh < alpha) * (cutQTL < q))] = np.nan
        return q
    plt.plot(alpha, ANTILAMBDA_CUT_LOW(alpha), label="Anti-Lambda Cut", color="red", linewidth = 4)

    # Gamma cuts
    def GAMMA_CUT1(alpha):
        return cutQTG * np.ones_like(alpha)

    def GAMMA_CUT2(alpha):
        return cutQTG2 * np.ones_like(alpha)

    def GAMMA_CUT_REGION(alpha):
        region = np.full_like(alpha, np.nan)
        mask1 = (np.abs(alpha) < cutAlphaG)
        mask2 = (cutAlphaGLow < np.abs(alpha)) & (np.abs(alpha) < cutAlphaGHigh)
        region[mask1] = cutQTG
        region[mask2] = cutQTG2
        return region

    plt.plot(alpha, GAMMA_CUT1(alpha), label="Gamma Cut 1", color="purple", linestyle="--", linewidth=4)
    plt.plot(alpha, GAMMA_CUT2(alpha), label="Gamma Cut 2", color="orange", linestyle="--", linewidth=4)
    plt.plot(alpha, GAMMA_CUT_REGION(alpha), label="Gamma Region", color="green", linestyle="-", linewidth=4)

###

## Loading data and models

particles = CONFIG['particle_info']['particles']
masses = CONFIG['particle_info']['masses']
LABELS_X = CONFIG['createTrainingDatasetOptions']['labels_x']
LABELS_Y = CONFIG['createTrainingDatasetOptions']['labels_y']
cload = load_tree()
TTree = cload.print_trees(dir_tree)
mode = args.loading_mode
import_labels = [*LABELS_Y, *LABELS_X, 'fPidIndex','fRunNumber']
print(f"[DEBUG]: Import labels are {import_labels}")
if mode=="full":
    labels, fit_data = cload.load(use_vars=import_labels, limit = 100, path=dir_tree, load_latest=True, verbose=True)
else:
    v0_labels, v0_data = cload.load(use_vars=import_labels, path=dir_tree, limit = 10000000, key="O2V0Tree", load_latest=True, verbose=True)

    ### Armenteros Podolanski selection
    alphaQt_l, alphaQt_d = cload.load(use_vars=["fAlphaV0", "fQtV0"], path=dir_tree, limit = 10000000, key="O2V0Tree", load_latest=True, verbose=True)
    particle_type = checkV0(alphaQt_d[:,alphaQt_l=="fAlphaV0"].flatten(), alphaQt_d[:,alphaQt_l=="fQtV0"].flatten(), **cut_dict) ### Returns boolean list of shape (4, n)
    mask_accept_V0 = np.sum(particle_type, axis=0).astype(bool)

    fig = plt.figure(figsize=(16, 10))
    plt.hist2d(alphaQt_d[:,alphaQt_l=="fAlphaV0"].flatten(), alphaQt_d[:,alphaQt_l=="fQtV0"].flatten(), bins=(500,500), range = ((-1.05,1.05), (0,0.25)), cmap = plt.cm.jet, norm=mcolors.LogNorm())
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$q_T$ (GeV/c)")
    plt.text(0,0.18, horizontalalignment='center', verticalalignment='center', fontsize=30, s=r"K$^{S}_0$", c="white")
    plt.text(-0.7,0.07, horizontalalignment='center', verticalalignment='center', fontsize=35, s=r"$\overline{\Lambda}$", c="white")
    plt.text(0.7,0.07, horizontalalignment='center', verticalalignment='center', fontsize=35, s=r"$\Lambda$", c="white")
    plt.colorbar(aspect=30, pad=0.01)
    plot_cuts(**cut_dict)
    plt.savefig(os.path.join(plot_path, "ArmenterosPodolanski_LHC24ar.pdf"), bbox_inches='tight')

    del alphaQt_d, alphaQt_l

    tpctof_labels, tpctof_data = cload.load(use_vars=import_labels, path=dir_tree, limit = 10000000, key="O2tpctofTree", load_latest=True, verbose=True)

    labels = tpctof_labels
    fit_data = v0_data[mask_accept_V0]
    fit_data = np.vstack((fit_data, tpctof_data))
    del tpctof_data

# Normalize fFT0Occ by a factor of 60000
ft0occ_index = np.where(labels == 'fFt0Occ')[0][0]  # Locate the index of fFT0Occ in labels
fit_data[:, ft0occ_index] /= 60000

if CONFIG['dataset']['HadronicRate'].strip().lower() == "true":
    print(f"[DEBUG]: Using Hadronic Rate in CreateDataset and normalise HadronicRate branch")
    print("[DEBUG]: Hadronic Rate values will be normalised by 50")
    # Normalize fHadronicRate by a factor of 50
    fHadronicRate_index = np.where(labels == 'fHadronicRate')[0][0]  # Locate the index of fHadronicRate in labels
    fit_data[:, fHadronicRate_index] /= 50

samplesize = int(CONFIG['createTrainingDatasetOptions']['samplesize'])
print(f"samplesize is {samplesize}")

if len(fit_data) >= samplesize:
    ### Downsampling to defined sample size
    keep = samplesize/len(fit_data) # Keep that many percent of the original data: Here keeping 60 mio., aribtrary but reasonable
    mask_downsample = np.random.uniform(low=0.0, high=1.0, size=len(fit_data)) < keep
    fit_data = fit_data[mask_downsample]

fig = plt.figure(figsize=(16,9))
x_space = np.logspace(-1., 1., 20*8)
plt.hist2d(fit_data[:,labels=="fTPCInnerParam"].flatten(), fit_data[:,labels=="fTPCSignal"].flatten(), bins=(x_space, np.arange(1,200,0.1)), range=[[-1.,2.],[1.,200.]], cmap=cm.jet, norm=mcolors.LogNorm())
plt.xscale("log")
plt.grid()
plt.savefig(os.path.join(plot_path, "initial_dEdx_vs_p.png"))
print("Saved initial dE/dx vs. p plot.")

### Trending of MIP vs. runnumber

print("Processing trending of MIP pions vs. runnumber...")

pions_at_mip = ((np.abs(fit_data[:, labels=="fTPCInnerParam"] - 0.4) < 0.01) * (np.abs(fit_data[:, labels=="fMass"] - 0.13957) < 0.0001)).flatten()
run_numbers = np.sort(np.unique(fit_data[:, labels=="fRunNumber"]))
gaussian_fits = list()
for run in run_numbers:
    print("Run:", int(run))
    run_mask = (fit_data[:, labels=="fRunNumber"] == run).flatten() * pions_at_mip
    run_data = fit_data[run_mask]
    for mult in tqdm(np.arange(0., 3., 0.1)):
        mask = (np.abs(run_data[:, labels=="fNormMultTPC"] - mult) < 0.1).flatten()
        sliced_data = run_data[mask]
        if np.sum(mask) < 1000:
            continue
        hist = np.histogram(sliced_data[:,labels=="fTPCSignal"].flatten(), bins=100, density=True)
        bin_center = (hist[1][1:] + hist[1][:-1]) / 2
        p0 = [1./(np.sqrt(2*np.pi)*0.07*np.average(bin_center, weights=hist[0])),np.average(bin_center, weights=hist[0]),0.07*np.average(bin_center, weights=hist[0])]
        coeff, var_matrix = sc.optimize.curve_fit(gauss, bin_center, hist[0], p0=p0)
        gaussian_fits.append([run, mult, *coeff[1:], np.sum(mask)])
gaussian_fits = np.array(gaussian_fits)

if len(gaussian_fits):
    mapper = dict()
    for i, run in enumerate(np.sort(np.unique(run_numbers))):
        mapper[run] = i
    index_x = list()
    for run in gaussian_fits[:,0]:
        index_x.append(mapper[run])

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(15, 15), gridspec_kw={'hspace': 0})

    scatter0 = axs[0].scatter(index_x, gaussian_fits[:,2], c=gaussian_fits[:,1], cmap="jet")
    cbar0 = fig.colorbar(scatter0, ax=axs[0])
    cbar0.set_label("norm. multiplicity", fontsize=18)
    cbar0.ax.tick_params(labelsize=15)
    axs[0].set_xlabel("runnumber", fontsize=18)
    axs[0].set_ylabel("dE/dx gaussian fit of MIP pions", fontsize=18)
    axs[0].grid()

    scatter1 = axs[1].scatter(index_x, gaussian_fits[:,2], c=gaussian_fits[:,3], cmap="hot")
    cbar1 = fig.colorbar(scatter1, ax=axs[1])
    cbar1.set_label("gaussian sigma", fontsize=18)
    cbar1.ax.tick_params(labelsize=15)
    axs[1].set_xlabel("runnumber", fontsize=18)
    axs[1].set_ylabel("dE/dx gaussian fit of MIP pions", fontsize=18)
    axs[1].grid()

    axs[2].set_xticks(np.arange(0, len(np.unique(gaussian_fits[:,0])), 1))
    axs[2].set_xticklabels(np.unique(gaussian_fits[:,0].astype(int)), rotation=90, fontsize=15)  # Set fontsize to 12
    scatter2 = axs[2].scatter(index_x, gaussian_fits[:,2], c=gaussian_fits[:,-1], cmap="inferno", norm=mcolors.LogNorm())
    cbar2 = fig.colorbar(scatter2, ax=axs[2])
    cbar2.set_label("#points in fit", fontsize=18)
    cbar2.ax.tick_params(labelsize=15)
    axs[2].set_xlabel("runnumber", fontsize=18)
    axs[2].set_ylabel("dE/dx gaussian fit of MIP pions", fontsize=18)
    axs[2].grid()

    plt.savefig(os.path.join(plot_path, "mip_pi_trending.png"))
else:
    print("No data for MIP trending. Skipping...")


### Index reordering

reorder_index = []
for lab in import_labels:
    reorder_index.append(np.where(labels==lab)[0][0])
reorder_index = np.array(reorder_index)
fit_data = fit_data[:,reorder_index]
labels = labels[reorder_index]

def gausscale(x, mean, sigma, scale):
    return (scale/0.4)*np.exp(-0.5*((x-mean)/sigma)**2)

def linear(x, a, b):
    return a*x + b

def gausgauslin(x, mu1, sigma1, scale1, mu2, sigma2, scale2, a, b):
    return gausscale(x, mu1, sigma1, scale1) + gausscale(x, mu2, sigma2, scale2) + (a*x + b)

def gauslin(x, mu1, sigma1, scale1, a, b):
    return gausscale(x, mu1, sigma1, scale1) + (a*x + b)


## Training data selection

### Kinematic selections
new_data = fit_data
# Initial 3 sigma selection
sigma_threshold = int(CONFIG['createTrainingDatasetOptions']['sigmarange'])
full_mask = np.zeros(len(new_data))

for i, m in enumerate(np.sort(np.unique(new_data.T[labels=='fMass']))):
    mask = (new_data.T[labels=='fMass'].flatten() == m)
    ratio_data = (new_data.T[labels=='fTPCSignal'].flatten()[mask]*new_data.T[labels=='fInvDeDxExpTPC'].flatten()[mask] - 1.)/0.07
    full_mask[mask] = np.abs(ratio_data)<sigma_threshold

new_data = new_data[full_mask.astype(bool)]

def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def selector(X, Y, rangeX, rangeY, bins_sigma_mean = 200, p0 = [1.,0,0.1], use_gauss=False):

    X = np.log10(X)
    rangeX = np.log10(rangeX)

    binned_mean = sc.stats.binned_statistic(X, Y, statistic='mean', bins=bins_sigma_mean, range=(rangeX[0],rangeX[1]))
    binned_sigma = sc.stats.binned_statistic(X, Y, statistic='std', bins=bins_sigma_mean, range=(rangeX[0],rangeX[1]))

    bins = np.logspace(rangeX[0], rangeX[1], bins_sigma_mean+1)

    if use_gauss:
        for j in range(bins_sigma_mean):
            try:
                hist1d = np.histogram(Y[(10**X>bins[j]) * (10**X<bins[j+1])], bins_sigma_mean, range=rangeY, density=False)
                if isinstance(p0, str):
                    p0 = [1./(np.sqrt(2*np.pi)*binned_sigma[0][j]),binned_mean[0][j],binned_sigma[0][j]]
                bin_centers = (hist1d[1][:-1] + hist1d[1][1:])/2
                coeff, var_matrix = sc.optimize.curve_fit(gauss, bin_centers, hist1d[0]/np.sum(hist1d[0]), p0=p0, bounds = ((-np.inf,-1.,0.),(np.inf,1.,np.inf)))
                if((coeff[1] < 0.5) * (coeff[2] > -0.5) * (coeff[2] < 1.5) * (coeff[2] > 0.7)):
                    binned_mean[0][j], binned_sigma[0][j] = coeff[1], coeff[2]
            except Exception as e:
                print(e)
                continue

    x_ax = np.linspace(rangeX[0],rangeX[1], bins_sigma_mean)
    mask_poly = (x_ax>rangeX[0]) * (x_ax<rangeX[1])
    poly_mean = np.polyfit(x_ax[mask_poly],np.array(binned_mean[0])[mask_poly],deg=13)
    poly_sigma = np.polyfit(x_ax[mask_poly],(np.array(binned_mean[0])+np.array(binned_sigma[0]))[mask_poly],deg=13)

    return (np.abs((Y - np.polyval(poly_mean,X))/(np.polyval(poly_sigma,X) - np.polyval(poly_mean, X))) < sigma_threshold), poly_mean, poly_sigma, binned_mean, binned_sigma
### Excluding outside 3 sigma range for indiv. particle species

momentum_ranges = {
    "Electrons": eval(args.cut_electrons),
    "Pions": eval(args.cut_pions),
    "Kaons": eval(args.cut_kaons),
    "Protons": eval(args.cut_protons),
    "Deuterons": eval(args.cut_deuterons),
    "Tritons": eval(args.cut_tritons)
}

momentum_ranges_array = np.array(list(momentum_ranges.values()))
p_cut = 10**momentum_ranges_array

rangeY = [-1.,1.]
bins_sigma_mean = 200

x_space = np.logspace(-1., 1.5, 20*8)
y_space = np.linspace(-5, 5, 20*8)

collect_data = 0

print("Processing data...")

for i, m in enumerate(tqdm(np.sort(np.unique(new_data.T[labels=='fMass']))[:4])):

    mask = (new_data.T[labels=='fMass'].flatten() == m)
    print(np.sum(mask))

    X = new_data[:,labels=="fTPCInnerParam"].flatten()[mask]
    Y = (new_data[:,labels=='fTPCSignal'].flatten()[mask]*new_data[:,labels=='fInvDeDxExpTPC'].flatten()[mask] - 1.)/0.07

    new_mask, poly_mean, poly_sigma, binned_mean, binned_sigma = selector(X, Y, 10**momentum_ranges_array[i], rangeY, bins_sigma_mean, p0 = "meansigma")

    test_data = new_data[mask]

    if i == 0:
        collect_data = test_data[new_mask * (test_data[:,labels=='fTPCInnerParam'].flatten() > p_cut[i][0]) * (test_data[:,labels=='fTPCInnerParam'].flatten() < p_cut[i][1])]# + ((test_data[:,labels=='fTPCInnerParam'].flatten() >0.93) * (test_ratio > -1.2) * (test_ratio < 2.7))]
    elif i == 1:
        collect_data = np.vstack((collect_data, test_data[new_mask * (test_data[:,labels=='fTPCInnerParam'].flatten() > p_cut[i][0]) * (test_data[:,labels=='fTPCInnerParam'].flatten() < p_cut[i][1])]))
    elif i == 2:
        # test_data = new_data[new_data.T[labels=='fMass'].flatten() == m]
        # test_ratio = (test_data[:,labels=='fTPCSignal'].flatten()*test_data[:,labels=='fInvDeDxExpTPC'].flatten() - 1)/0.05
        collect_data = np.vstack((collect_data, test_data[new_mask * (test_data[:,labels=='fTPCInnerParam'].flatten() > p_cut[i][0]) * (test_data[:,labels=='fTPCInnerParam'].flatten() < p_cut[i][1])]))
    elif i == 3:
        collect_data = np.vstack((collect_data, test_data[new_mask * (test_data[:,labels=='fTPCInnerParam'].flatten() > p_cut[i][0]) * (test_data[:,labels=='fTPCInnerParam'].flatten() < p_cut[i][1])]))
    elif i == 4:
        collect_data = np.vstack((collect_data, test_data[new_mask * (test_data[:,labels=='fTPCInnerParam'].flatten() > p_cut[i][0]) * (test_data[:,labels=='fTPCInnerParam'].flatten() < p_cut[i][1])]))
    elif i == 5:
        collect_data = np.vstack((collect_data, test_data[new_mask * (test_data[:,labels=='fTPCInnerParam'].flatten() > p_cut[i][0]) * (test_data[:,labels=='fTPCInnerParam'].flatten() < p_cut[i][1])]))
    else:
        print("This mass (" + str(m) + ") is not supported!")

    fig = plt.figure(figsize=(16,9))
    plt.hist2d(X, Y, bins = (x_space, y_space), cmap=cm.jet, norm=mcolors.LogNorm())
    plt.colorbar()
    plt.errorbar(np.logspace(momentum_ranges_array[i][0], momentum_ranges_array[i][1], bins_sigma_mean), binned_mean[0], yerr=binned_sigma[0], xerr=0, fmt='.', capsize=2., c='black', ls='none', elinewidth=1.)
    plt.plot(10**np.linspace(momentum_ranges_array[i][0],momentum_ranges_array[i][1], 1000),np.polyval(poly_mean,np.linspace(momentum_ranges_array[i][0],momentum_ranges_array[i][1], 1000)), linewidth=3., label='Poly. fit (deg = 13), mean')
    plt.plot(10**np.linspace(momentum_ranges_array[i][0],momentum_ranges_array[i][1], 1000),np.polyval(poly_sigma,np.linspace(momentum_ranges_array[i][0],momentum_ranges_array[i][1], 1000)), linewidth=3.,  label='Poly. fit (deg = 13), sigma')
    plt.plot(10**np.linspace(momentum_ranges_array[i][0],momentum_ranges_array[i][1], 1000),
            3*(np.polyval(poly_sigma,np.linspace(momentum_ranges_array[i][0],momentum_ranges_array[i][1], 1000))-np.polyval(poly_mean,np.linspace(momentum_ranges_array[i][0],momentum_ranges_array[i][1], 1000))) + np.polyval(poly_mean,np.linspace(momentum_ranges_array[i][0],momentum_ranges_array[i][1], 1000)),
            c="black", linewidth=3.,  label='Poly. fit (deg = 13), 3 sigma')
    plt.plot(10**np.linspace(momentum_ranges_array[i][0],momentum_ranges_array[i][1], 1000),
            -3*(np.polyval(poly_sigma,np.linspace(momentum_ranges_array[i][0],momentum_ranges_array[i][1], 1000))-np.polyval(poly_mean,np.linspace(momentum_ranges_array[i][0],momentum_ranges_array[i][1], 1000))) + np.polyval(poly_mean,np.linspace(momentum_ranges_array[i][0],momentum_ranges_array[i][1], 1000)),
            c="black", linewidth=3.,  label='Poly. fit (deg = 13), -3 sigma')

    plt.axvline(p_cut[i][0], c='black')
    plt.axvline(p_cut[i][1], c='black')

    plt.xscale('log')
    plt.xlabel('p [GeV/c]')
    plt.ylabel('TPC Nσ, σ=7%')
    plt.savefig(os.path.join(plot_path, "dEdx_vs_p_selection_{0}.png".format(particles[np.where(np.abs(np.array(masses) - m)<0.001)[0][0]])))
    print("Saved dE/dx vs. p selection plot for {0}.".format(particles[np.where(np.abs(np.array(masses) - m)<0.001)[0][0]]))

new_data = collect_data

### Down- / Oversampling
import random

print("Data-points before: ", np.shape(new_data)[0])

percentages = []
for i, m in enumerate(np.sort(np.unique(new_data.T[labels=='fMass']))):
    percentages.append(np.sum(new_data.T[labels=='fMass'].flatten() == m)*100/np.shape(new_data)[0])

desired_size = int(CONFIG['createTrainingDatasetOptions']['samplesize'])
num_particles = np.shape(np.unique(fit_data.T[labels=='fMass']))[0]

if len(np.unique(new_data.T[labels=='fMass']))==5:
    percentiles = [0.21,0.21,0.21,0.21,0.16]
elif len(np.unique(new_data.T[labels=='fMass']))==4:
    percentiles = [0.25, 0.25, 0.25, 0.25]
else:
    percentiles = [0.18,0.18,0.18,0.18,0.14,0.14]

for i, m in enumerate(np.sort(np.unique(new_data.T[labels=='fMass']))):
    # if i==2:
    #     mask = (new_data.T[labels=='fMass'].flatten() == m)
    #     new_data = np.vstack((new_data,new_data[mask]))
    # if i==1:
    #     mask = (new_data.T[labels=='fMass'].flatten() == m)
    #     keep_others = (~mask).copy()
    #     mask = mask * (np.random.random(np.shape(mask))>0.2)
    #     new_data = new_data[keep_others+mask]

    mask = (new_data.T[labels=='fMass'].flatten() == m)

    ### Downsample
    if np.sum(mask)>desired_size*percentiles[i]:
        keep_others = (~mask).copy()
        mask = mask * (np.random.random(np.shape(mask)[0])<((desired_size*percentiles[i])/np.sum(mask)))
        new_data = new_data[keep_others+mask]
    ### Upsample
    else:
        randint = np.random.randint(0, np.sum(mask)-1, int(np.floor(desired_size*percentiles[i]-np.sum(mask))))
        new_data = np.vstack((new_data, new_data[mask][randint]))


print("Data-points after: ", np.shape(new_data)[0], "\n")

for i, m in enumerate(np.sort(np.unique(new_data.T[labels=='fMass']))):
    print(particles[np.where(np.abs(np.array(masses) - m)<0.001)[0][0]], ": ", np.sum(new_data.T[labels=='fMass'].flatten() == m)*100/np.shape(new_data)[0], "%")

### Export data to file
cload = load_tree()
cload.export_to_tree(output_path, np.array(labels).astype(str), new_data, overwrite=True)

print("Exported data to: ", output_path)