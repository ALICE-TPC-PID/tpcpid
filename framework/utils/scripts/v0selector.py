import os
import sys

import numpy as np
import scipy as sc
from scipy import stats
import matplotlib.pyplot as plt

from tqdm import tqdm

import matplotlib as mpl
from matplotlib import cm
import matplotlib.colors as mcolors
import pandas as pd
import argparse

sys.path.append("../../../Neural-Network-Class/NeuralNetworkClasses")
from extract_from_root import *

# Argument parser
parser = argparse.ArgumentParser(description="V0 Selector Configuration")
parser.add_argument("-i", "--input", type=str, default="v0.root", help="Input ROOT file")
parser.add_argument("-o", "--output", type=str, default="v0_2.root", help="Output ROOT file")
parser.add_argument("-es", "--export-size", type=int, default=100000000, help="Export size for each file. They can then be concatenated using hadd")

parser.add_argument("--cutAlphaG", type=float, default=0.4, help="Gamma cut: AlphaG")
parser.add_argument("--cutQTG", type=float, default=0.006, help="Gamma cut: QTG")
parser.add_argument("--cutAlphaGLow", type=float, default=0.4, help="Gamma cut: AlphaGLow")
parser.add_argument("--cutAlphaGHigh", type=float, default=0.8, help="Gamma cut: AlphaGHigh")
parser.add_argument("--cutQTG2", type=float, default=0.006, help="Gamma cut: QTG2")
parser.add_argument("--cutQTK0SLow", type=float, default=0.1075, help="K0S cut: QTK0SLow")
parser.add_argument("--cutQTK0SHigh", type=float, default=0.215, help="K0S cut: QTK0SHigh")
parser.add_argument("--cutAPK0SLow", type=float, default=0.199, help="K0S cut: APK0SLow")
parser.add_argument("--cutAPK0SHigh", type=float, default=0.8, help="K0S cut: APK0SHigh")
parser.add_argument("--cutAPK0SHighTop", type=float, default=1.0, help="K0S cut: APK0SHighTop")
parser.add_argument("--cutQTL", type=float, default=0.03, help="Lambda cut: QTL")
parser.add_argument("--cutAlphaLLow", type=float, default=0.35, help="Lambda cut: AlphaLLow")
parser.add_argument("--cutAlphaLLow2", type=float, default=0.53, help="Lambda cut: AlphaLLow2")
parser.add_argument("--cutAlphaLHigh", type=float, default=0.7, help="Lambda cut: AlphaLHigh")
parser.add_argument("--cutAPL1", type=float, default=0.107, help="Lambda cut: APL1")
parser.add_argument("--cutAPL2", type=float, default=-0.69, help="Lambda cut: APL2")
parser.add_argument("--cutAPL3", type=float, default=0.5, help="Lambda cut: APL3")
parser.add_argument("--cutAPL1Low", type=float, default=0.091, help="Lambda cut: APL1Low")
parser.add_argument("--cutAPL2Low", type=float, default=-0.69, help="Lambda cut: APL2Low")
parser.add_argument("--cutAPL3Low", type=float, default=0.156, help="Lambda cut: APL3Low")

args = parser.parse_args()

print("Loading data...")

COMMON_BRANCHES = ['fTPCInnerParam', 'fTgl', 'fSigned1Pt', 'fMass', 'fNormMultTPC', 'fNormNClustersTPC', 'fFt0Occ', 'fTPCSignal', 'fInvDeDxExpTPC', 'fRunNumber']
V0_BRANCHES = ['fAlphaV0', 'fQtV0']
cload = load_tree()
labels_V0, fit_data_V0 = cload.load(use_vars=[*COMMON_BRANCHES, *V0_BRANCHES], key="O2tpcskimv0tree",
                              path=args.input, load_latest=True, verbose=True)

### V0 cleaning
cut_dict = {
    # Gamma cuts
    "cutAlphaG": args.cutAlphaG,
    "cutQTG": args.cutQTG,
    "cutAlphaGLow": args.cutAlphaGLow,
    "cutAlphaGHigh": args.cutAlphaGHigh,
    "cutQTG2": args.cutQTG2,

    # K0S cuts
    "cutQTK0SLow": args.cutQTK0SLow,
    "cutQTK0SHigh": args.cutQTK0SHigh,
    "cutAPK0SLow": args.cutAPK0SLow,
    "cutAPK0SHigh": args.cutAPK0SHigh,
    "cutAPK0SHighTop": args.cutAPK0SHighTop,

    # Lambda & Anti-Lambda cuts
    "cutQTL": args.cutQTL,
    "cutAlphaLLow": args.cutAlphaLLow,
    "cutAlphaLLow2": args.cutAlphaLLow2,
    "cutAlphaLHigh": args.cutAlphaLHigh,
    "cutAPL1": args.cutAPL1,
    "cutAPL2": args.cutAPL2,
    "cutAPL3": args.cutAPL3,
    "cutAPL1Low": args.cutAPL1Low,
    "cutAPL2Low": args.cutAPL2Low,
    "cutAPL3Low": args.cutAPL3Low
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

print("Performing V0 selection...")

particle_type = checkV0(fit_data_V0[:,labels_V0=="fAlphaV0"].flatten(), fit_data_V0[:,labels_V0=="fQtV0"].flatten(), **cut_dict) ### Returns boolean list of shape (4, n)
mask_accept_V0 = np.sum(particle_type, axis=0).astype(bool)
fit_data_V0 = fit_data_V0[mask_accept_V0]

print("Rejected: " + str(int(np.sum(mask_accept_V0 == False))) + " / " + str(len(fit_data_V0)) + " (" + "{:.4f}".format(int(np.sum(mask_accept_V0 == False)) * 100 / len(fit_data_V0)) + "%)")

fontsize_axislabels = 17
fig = plt.figure(figsize=(16, 10))
plt.hist2d(fit_data_V0[:,labels_V0=="fAlphaV0"].flatten(), fit_data_V0[:,labels_V0=="fQtV0"].flatten(), bins=(500,500), range = ((-1.05,1.05), (0,0.25)), cmap = plt.cm.jet, norm=mcolors.LogNorm(), zorder=3)
plt.xlabel(r"$\alpha$", fontsize=fontsize_axislabels)
plt.ylabel(r"$q_T$ (GeV/c)", fontsize=fontsize_axislabels)
plt.text(0,0.02, horizontalalignment='center', verticalalignment='center', fontsize=35, s=r"$\gamma$", c="black", zorder=2)
plt.text(0,0.18, horizontalalignment='center', verticalalignment='center', fontsize=30, s=r"K$^{S}_0$", c="black", zorder=2)
plt.text(-0.7,0.07, horizontalalignment='center', verticalalignment='center', fontsize=35, s=r"$\overline{\Lambda}$", c="black", zorder=2)
plt.text(0.7,0.07, horizontalalignment='center', verticalalignment='center', fontsize=35, s=r"$\Lambda$", c="black", zorder=2)
plt.colorbar(aspect=30, pad=0.01)
plt.grid(zorder=1)
plot_cuts(**cut_dict)
plt.savefig(os.path.join(os.path.dirname(args.output), "ArmenterosPodolanski.pdf"), bbox_inches='tight')

accept_common_branches = [b in COMMON_BRANCHES for b in labels_V0]
fit_data_V0 = fit_data_V0[:,accept_common_branches]

print("Loading full data and exporting...")

tpctof_labels, tpctof_data = cload.load(use_vars=COMMON_BRANCHES, key="O2tpctofskimtree", path=args.input, load_latest=True, verbose=True)
tpctof_data = np.vstack((fit_data_V0, tpctof_data))

for i in range(np.ceil(len(tpctof_data)/float(args.export_size)).astype(int)):
    start = i * args.export_size
    end = (i + 1) * args.export_size
    if end > len(tpctof_data):
        end = len(tpctof_data)
    cload.export_to_tree(args.output.replace(".root", "_" + str(i) + ".root"), np.array(COMMON_BRANCHES).astype(str), tpctof_data[start:end, :], overwrite=True)

print("Done.")