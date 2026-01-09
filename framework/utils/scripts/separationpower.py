import sys
import os

sys.path.append("/lustre/alice/users/csonnab/TPC/o2-tpc-pid/Neural-Network-Class/NeuralNetworkClasses")

from tqdm import tqdm
from extract_from_root import *
from NN_class import *

import numpy as np
import scipy as sc
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import onnxruntime as ort

from scipy.optimize import curve_fit

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

import matplotlib as mpl
from matplotlib import cm
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap

mpl.rcParams['figure.figsize'] = (16, 9)
plt.rcParams.update({'text.latex.preamble' : r'\usepackage{amsmath}'})

### Please adjust the following variables according to your needs

output_dir = "/lustre/alice/users/csonnab/TPC/o2-tpc-pid/Training-Neural-Networks/misc/notebooks/Test"
nn_path = "/lustre/alice/users/jwitte/tpcpid/o2-tpcpid-parametrisation/Training-Neural-Networks/output/LHC24/ar_pass1_NN/default_default/networks/network_full/net_onnx_full.onnx"
data_file = "/lustre/alice/users/marin/R3A/TPCTrees/TPCTrees_LHC24ar_250110/AO2D_merge_LHC24ar.root"
bb_params = [0.228007,3.93226,0.0122857,2.26946,0.861199,50,2.3]

### -------------------------------------------------

### Functions start from here. No adjustment needed

fontsize_axislabels = 17
plot_bins = 100

LABELS_X = ['fTPCInnerParam', 'fTgl', 'fSigned1Pt', 'fMass', 'fNormMultTPC', 'fNormNClustersTPC', 'fFt0Occ']
LABELS_Y = ['fTPCSignal', 'fInvDeDxExpTPC']

ort_sess_full = ort.InferenceSession(nn_path)

cload = load_tree()
labels_all, fit_data_all = cload.load(data_file, use_vars = [*LABELS_X, *LABELS_Y, 'fHadronicRate'], load_latest=True, verbose = True)

particles = ['Electrons', 'Muons', 'Pions', 'Kaons', 'Protons', 'Deuteron', 'Triton', r'Helium3']
particle_labels = ['e', r'$\mu$', r'$\pi$', r'$K$', r'$p$', r'$d$', r'$t$', r'$^3$He']
masses = [0.000510998950, 0.1056583755, 0.13957039, 0.493677, 0.93827208816, 1.875613115, 2.8089211, 2.8083916]
charges = [1.,1.,1.,1.,1.,1.,1.,2.]
dict_particles_masses = dict(zip(particles, masses))

reorder_index = []
for lab in [*LABELS_Y, *LABELS_X]:
    reorder_index.append(np.where(labels_all==lab)[0][0])
reorder_index = np.array(reorder_index)
fit_data = fit_data_all[:,reorder_index]
labels = labels_all[reorder_index]
mask_X = []
mask_y = []
for l in labels:
    mask_X.append(l in LABELS_X)
    mask_y.append(l in LABELS_Y)

def network(data, ort_session=ort_sess_full):
    return np.array(ort_session.run(None, {'input': (torch.tensor(data).float()).numpy()}))

def BetheBlochAleph(bg, params):
    beta = bg/np.sqrt(1.+ bg*bg)
    aa   = beta**params[3]
    bb   = bg**(-params[4])
    bb   = np.log(params[2]+bb)
    charge_factor = params[5]         # params[5] = mMIP, params[6] = mChargeFactor #usually its the other way around. Here its just for simplicity to copy it directly from the google sheet
    final = (params[1]-aa-bb)*params[0]*charge_factor/aa
    return final


fit_dict = dict()

def separation_power(labels_loc, fit_data_loc, useNN=0, useMassAssumption=0, momentumSelection=[0.3,0.4],
                     gauss_labels = {
                         0: ["electrons", "pions"],
                         2: ["pions", "electrons"]
                     },
                     y_ranges = {
                        0: [-10.,3.],
                        2: [-3.,10.]
                     },
                     initial_params = {
                         0: [None,0.,1.,None,-3.,1.],
                         2: [None,0.,1.,None,4.,1.]
                     },
                     gauss_bounds = {
                         0: ([0.,-2.,0.,0.,-10.,0.],[10.,2.,3.,10.,-3.,3.]),
                         2: ([0.,-2.,0.,0.,3.,0.],[10.,2.,3.,10.,10.,3.])
                     }):

    plot_labels = {
        "pions": ["Pions", "red"],
        "electrons": ["Electrons", "orange"]
    }

    # os.makedirs(output_dir + "/SeparationPower/debug", exist_ok=True)
    os.makedirs(output_dir + "/SeparationPower", exist_ok=True)

    ### usemMassAssumption is the index in the masses array: 0 = electrons, 2 = pions

    def separation_power(mu1, mu2, sigma1, sigma2):
        return 2*np.abs(mu1-mu2)/(sigma1+sigma2)

    fig = plt.figure(figsize=(16,9))
    y_bins = np.linspace(*(y_ranges[useMassAssumption]),100)

    input = fit_data_loc.copy()
    selection = (input[:,labels_loc=='fTPCInnerParam'].flatten() > 0.3) * (input[:,labels_loc=='fTPCInnerParam'].flatten() < 0.4)
    input = input[selection]
    input[:,labels_loc=='fInvDeDxExpTPC'] = (1./BetheBlochAleph(fit_data_loc[selection,labels_loc=='fTPCInnerParam'].flatten()/masses[useMassAssumption], params = bb_params)).reshape(-1,1)
    input[:,labels_loc=='fMass'] = masses[useMassAssumption]
    input[:,labels_loc=='fFt0Occ'] /= 60000.

    if useNN:
        net_out = network(input[:,mask_X],ort_session=ort_sess_full)[0]
        temp=net_out.T[1].flatten()-net_out.T[0].flatten()
        temp[temp==0]=1e-9
        output = (((input[:,labels_loc=='fTPCSignal'].flatten()*input[:,labels_loc=='fInvDeDxExpTPC'].flatten()) - net_out.T[0].flatten()))/(temp)
    else:
        output = (input[:,labels_loc=='fTPCSignal'].flatten()*input[:,labels_loc=='fInvDeDxExpTPC'].flatten() - 1.)/0.1

    hist1d = np.histogram(output, bins=y_bins, density=True)
    initial_params[useMassAssumption][0] = np.max(hist1d[0])
    initial_params[useMassAssumption][3] = np.max(hist1d[0])
    popt, pcov = sc.optimize.curve_fit(double_gauss, y_bins[:-1], hist1d[0], p0=initial_params[useMassAssumption], bounds=gauss_bounds[useMassAssumption])
    plt.hist(output, bins=y_bins, histtype='step', lw=2, color="black", density=True)
    plt.plot(y_bins[:-1], double_gauss(y_bins[:-1], *popt), lw=1, label='Double gauss fit', c = "blue")
    plt.plot(y_bins[:-1], gauss(y_bins[:-1], popt[0], popt[1], popt[2]), lw=1, label=plot_labels[gauss_labels[useMassAssumption][0]][0] + " {:.3f}".format(popt[1]) + " $\pm$ " + "{:.3f}".format(popt[2]), c = plot_labels[gauss_labels[useMassAssumption][0]][1])
    plt.plot(y_bins[:-1], gauss(y_bins[:-1], popt[3], popt[4], popt[5]), lw=1, label=plot_labels[gauss_labels[useMassAssumption][1]][0] + " {:.3f}".format(popt[4]) + " $\pm$ " + "{:.3f}".format(popt[5]), c = plot_labels[gauss_labels[useMassAssumption][1]][1])
    plt.axvline(0, c='black', lw=1, ls='--', label="Optimal band center")
    sep_power = separation_power(popt[1], popt[4], popt[2], popt[5])

    plt.xlabel(r'N$\sigma$ (' + particles[useMassAssumption] + ')', fontsize=fontsize_axislabels)
    plt.ylabel('Density', fontsize=fontsize_axislabels)
    plt.legend(title="Separation power: " + "{:.3f}".format(sep_power))
    plt.grid()
    if useNN:
        plt.savefig(output_dir + '/SeparationPower/SeparationPower_' + particles[useMassAssumption] + 'MassHypothesis_NN.pdf', bbox_inches='tight')
    else:
        plt.savefig(output_dir + '/SeparationPower/SeparationPower_' + particles[useMassAssumption] + 'MassHypothesis_BB.pdf', bbox_inches='tight')
    plt.close()

    hist, edges = np.histogram(fit_data_all[:,labels_all=="fHadronicRate"].flatten(), bins=plot_bins, range=(np.min(fit_data_all[:,labels_all=="fHadronicRate"]), np.max(fit_data_all[:,labels_all=="fHadronicRate"])), density=True)
    trending_output = {
        "mean_1": list(),
        "sigma_1_up": list(),
        "sigma_1_low": list(),
        "mean_2": list(),
        "sigma_2_up": list(),
        "sigma_2_low": list(),
        "separation_p": list(),
        "x_points": list()
    }
    plot_labels_trending = [["mean " + plot_labels[gauss_labels[useMassAssumption][0]][0], "blue"],
                            ["sigma " + plot_labels[gauss_labels[useMassAssumption][0]][0], "deepskyblue"],
                            ["", "deepskyblue"],
                            ["mean " + plot_labels[gauss_labels[useMassAssumption][1]][0], "orange"],
                            ["sigma " + plot_labels[gauss_labels[useMassAssumption][1]][0], "red"],
                            ["", "red"],
                            ["separation power", "black"]]
    for i, edge in enumerate(edges[:-1]):
        try:
            hist1d = np.histogram(output[(fit_data_all[selection,labels_all=="fHadronicRate"].flatten() > edges[i]) * (fit_data_all[selection,labels_all=="fHadronicRate"].flatten() <= edges[i+1])], bins=y_bins, density=True)
            popt, pcov = sc.optimize.curve_fit(double_gauss, y_bins[:-1], hist1d[0], p0=initial_params[useMassAssumption], bounds=gauss_bounds[useMassAssumption])
            sp = separation_power(popt[1], popt[4], popt[2], popt[5])
            if sp < 10:
                trending_output["mean_1"].append(popt[1])
                trending_output["mean_1"].append(popt[1])

                trending_output["mean_2"].append(popt[4])
                trending_output["mean_2"].append(popt[4])

                trending_output["sigma_1_up"].append(popt[1] + popt[2])
                trending_output["sigma_1_up"].append(popt[1] + popt[2])
                trending_output["sigma_1_low"].append(popt[1] - popt[2])
                trending_output["sigma_1_low"].append(popt[1] - popt[2])

                trending_output["sigma_2_up"].append(popt[4] + popt[5])
                trending_output["sigma_2_up"].append(popt[4] + popt[5])
                trending_output["sigma_2_low"].append(popt[4] - popt[5])
                trending_output["sigma_2_low"].append(popt[4] - popt[5])

                trending_output["separation_p"].append(sp)
                trending_output["separation_p"].append(sp)

                trending_output["x_points"].append(edges[i])
                trending_output["x_points"].append(edges[i+1])

            else:
                trending_output["mean_1"].append(np.nan)
                trending_output["mean_1"].append(np.nan)

                trending_output["mean_2"].append(np.nan)
                trending_output["mean_2"].append(np.nan)

                trending_output["sigma_1_up"].append(np.nan)
                trending_output["sigma_1_up"].append(np.nan)
                trending_output["sigma_1_low"].append(np.nan)
                trending_output["sigma_1_low"].append(np.nan)

                trending_output["sigma_2_up"].append(np.nan)
                trending_output["sigma_2_up"].append(np.nan)
                trending_output["sigma_2_low"].append(np.nan)
                trending_output["sigma_2_low"].append(np.nan)

                trending_output["separation_p"].append(np.nan)
                trending_output["separation_p"].append(np.nan)

                trending_output["x_points"].append(edges[i])
                trending_output["x_points"].append(edges[i+1])


        except Exception as e:
            trending_output["mean_1"].append(np.nan)
            trending_output["mean_1"].append(np.nan)

            trending_output["mean_2"].append(np.nan)
            trending_output["mean_2"].append(np.nan)

            trending_output["sigma_1_up"].append(np.nan)
            trending_output["sigma_1_up"].append(np.nan)
            trending_output["sigma_1_low"].append(np.nan)
            trending_output["sigma_1_low"].append(np.nan)

            trending_output["sigma_2_up"].append(np.nan)
            trending_output["sigma_2_up"].append(np.nan)
            trending_output["sigma_2_low"].append(np.nan)
            trending_output["sigma_2_low"].append(np.nan)

            trending_output["separation_p"].append(np.nan)
            trending_output["separation_p"].append(np.nan)

            trending_output["x_points"].append(edges[i])
            trending_output["x_points"].append(edges[i+1])
            print(i, ":", e)

            # fig = plt.figure(figsize=(16,9))
            # plt.hist(output[(fit_data_all[selection,labels_all=="fHadronicRate"].flatten() > edges[i]) * (fit_data_all[selection,labels_all=="fHadronicRate"].flatten() <= edges[i+1])], bins=y_bins, histtype='step', lw=2, color="black", density=True)
            # plt.xlabel(r'N$\sigma$ (' + particles[useMassAssumption] + ')', fontsize=fontsize_axislabels)
            # plt.ylabel('Density', fontsize=fontsize_axislabels)
            # plt.grid()
            # if useNN:
            #     plt.savefig(output_dir + '/SeparationPower/debug/SeparationPower_' + particles[useMassAssumption] + 'MassHypothesis_NN' + str(i) + '.pdf', bbox_inches='tight')
            # else:
            #     plt.savefig(output_dir + '/SeparationPower/debug/SeparationPower_' + particles[useMassAssumption] + 'MassHypothesis_BB' + str(i) + '.pdf', bbox_inches='tight')
            # plt.close()

    fit_dict[particles[useMassAssumption] + ("_NN" if useNN else "_BB")] = trending_output
    # np.save(output_dir + '/SeparationPower/SeparationPower_trending_' + particles[useMassAssumption] + 'MassHypothesis_NN.npy', trending_output)
    print("Trending done! Plotting...")
    fig, axs = plt.subplots(3, 1, figsize=(16, 20), gridspec_kw={'height_ratios': [3, 1, 1], 'hspace': 0})

    # First plot
    for j, p in enumerate(trending_output.keys()):
        if p != "x_points" and p != "separation_p":
            axs[0].plot(trending_output["x_points"], trending_output[p], label=plot_labels_trending[j][0], c=plot_labels_trending[j][1])
    axs[0].set_xlabel("")  # Remove x-axis label from the first plot
    axs[0].set_ylabel(r'N$\sigma$ (' + particles[useMassAssumption] + ')', fontsize=fontsize_axislabels)
    axs[0].set_xlim(np.min(fit_data_all[:, labels_all == "fHadronicRate"]), np.max(fit_data_all[:, labels_all == "fHadronicRate"]))
    axs[0].legend()
    axs[0].grid()

    # Second plot (new plot for separation power)
    axs[1].plot(trending_output["x_points"], trending_output["separation_p"], label="Separation Power", c="black")
    axs[1].set_xlabel("")  # Remove x-axis label from the second plot
    axs[1].set_ylabel("Separation Power", fontsize=fontsize_axislabels)
    axs[1].set_xlim(np.min(fit_data_all[:, labels_all == "fHadronicRate"]), np.max(fit_data_all[:, labels_all == "fHadronicRate"]))
    axs[1].legend()
    axs[1].grid()

    # Third plot
    axs[2].hist(fit_data_all[:, labels_all == "fHadronicRate"].flatten(), bins=plot_bins,
                range=(np.min(fit_data_all[:, labels_all == "fHadronicRate"]),
                       np.max(fit_data_all[:, labels_all == "fHadronicRate"])), density=True)
    axs[2].set_xlabel("Hadronic rate", fontsize=fontsize_axislabels)
    axs[2].set_ylabel("Density", fontsize=fontsize_axislabels)
    axs[2].set_xlim(np.min(fit_data_all[:, labels_all == "fHadronicRate"]), np.max(fit_data_all[:, labels_all == "fHadronicRate"]))
    axs[2].grid()
    if useNN:
        plt.savefig(output_dir + '/SeparationPower/SeparationPower_trending_' + particles[useMassAssumption] + 'MassHypothesis_NN.pdf', bbox_inches='tight')
    else:
        plt.savefig(output_dir + '/SeparationPower/SeparationPower_trending_' + particles[useMassAssumption] + 'MassHypothesis_BB.pdf', bbox_inches='tight')
    plt.close()

fig = plt.figure(figsize=(16,9))
plt.hist(fit_data_all[:,labels_all=="fHadronicRate"].flatten(), bins=plot_bins, range=(np.min(fit_data_all[:,labels_all=="fHadronicRate"]), np.max(fit_data_all[:,labels_all=="fHadronicRate"])), density=True)
plt.xlim(np.min(fit_data_all[:,labels_all=="fHadronicRate"]), np.max(fit_data_all[:,labels_all=="fHadronicRate"]))
plt.xlabel("Hadronic rate [kHz]", fontsize=fontsize_axislabels)
plt.ylabel("Density", fontsize=fontsize_axislabels)
plt.grid()
plt.savefig(output_dir + '/SeparationPower/HadronicRate.pdf', bbox_inches='tight')
plt.close()

separation_power(labels, fit_data, useNN=1, useMassAssumption=0)
separation_power(labels, fit_data, useNN=0, useMassAssumption=0)
separation_power(labels, fit_data, useNN=1, useMassAssumption=2)
separation_power(labels, fit_data, useNN=0, useMassAssumption=2)

np.save(output_dir + '/SeparationPower/fit_dict.npy', fit_dict)