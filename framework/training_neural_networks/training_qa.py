"""
File: training_qa.py
Author: Christian Sonnabend
Email: christian.sonnabend@cern.ch
Date: 15/03/2024
"""

import sys
import os
import argparse
import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LogNorm
import scipy as sc
import json
import torch

import matplotlib as mpl

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="configuration.json", help="Path to the configuration file")
args = parser.parse_args()

with open(args.config, 'r') as config_file:
    CONFIG = json.load(config_file)
sys.path.append(CONFIG['settings']['framework'] + "/framework")
from base import *
from math_functions import *
from neural_network_class.NeuralNetworkClasses.extract_from_root import *
nnconfig = import_from_path(CONFIG["trainNeuralNetOptions"]["configuration"])
LOG = logger("training_qa")

########### Import the Neural Network class ###########

# Follow old-script behavior: read HadronicRate from config and cast to bool
HadronicRateBool = "fHadronicRate" in CONFIG['createTrainingDatasetOptions']['labels_x']
if HadronicRateBool:
    LOG.debug("Using Hadronic Rate option")

### Data preparation

## Loading data and models

particles                   = particle_info["particles"]
particle_labels             = particle_info["particle_labels"]
masses                      = particle_info["masses"]
charges                     = particle_info["charges"]
dict_particles_masses       = dict(zip(particles, masses))

### Neural Network

training_folder = CONFIG["output"]["general"]["training"] #general output dir for training
qa_dir          = CONFIG["output"]["trainNeuralNet"]["QApath"] #output directory for QA plots
data_path       = CONFIG["output"]["createTrainingDataset"]["training_data"]

LABELS_X = CONFIG["createTrainingDatasetOptions"]["labels_x"]
LABELS_Y = CONFIG["createTrainingDatasetOptions"]["labels_y"]

### General

jet_map = cm.jet(np.arange(cm.jet.N))
jet_map[:,-1] = np.linspace(0, 1, cm.jet.N)
jet_map_alpha = ListedColormap(jet_map)

fontsize_axislabels = 30
momentum = np.logspace(-2,3,1000)
cload = load_tree()
data = cload.load(use_vars=None, path=data_path, load_latest=True, verbose=True)
labels = np.array(data[0])
fit_data = data[1]
del data

reorder_index = []
for lab in [*LABELS_Y, *LABELS_X]:
    reorder_index.append(np.where(labels==lab)[0][0])
reorder_index = np.array(reorder_index)
fit_data = fit_data[:,reorder_index]
labels = labels[reorder_index]
mask_X = []
mask_y = []
for l in labels:
    mask_X.append(l in LABELS_X)
    mask_y.append(l in LABELS_Y)

X = fit_data[:,mask_X]
y = (fit_data[:,mask_y].T[0].flatten()*fit_data[:,mask_y].T[1].flatten())

sess_options = ort.SessionOptions()
sess_options.execution_mode  = ort.ExecutionMode.ORT_PARALLEL
sess_options.intra_op_num_threads = 0
sess_options.inter_op_num_threads = 10
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

ort_sess_mean = ort.InferenceSession(training_folder+"/networks/network_mean/net_onnx_mean.onnx", sess_options)
ort_sess_sigma = ort.InferenceSession(training_folder+"/networks/network_sigma/net_onnx_sigma.onnx", sess_options)
ort_sess_full = ort.InferenceSession(training_folder+"/networks/network_full/net_onnx_full.onnx", sess_options)

def network(data, ort_session=ort_sess_full):
    return np.array(ort_session.run(None, {'input': (torch.tensor(data).float()).numpy()}))

net_out = network(X,ort_session=ort_sess_full)[0] ### Precompute for speed


def QA2D_NSigma_vs_Var(i, mass, plot_against = 'fTPCInnerParam', log_x = True, range_hists = [[-1.,1.]]*6, fitted_particles=['Electrons', 'Pions', 'Kaons', 'Protons', 'Deuterons', 'Tritons'], bins_sig_mean=100, sigma_range = [-3.,3.], useNN = True, xlabel = r'p [GeV/c]', transform_x = lambda x: x, plot_mode="nsigma"):

    LOG.info("Creating histograms for " + fitted_particles[i] + " against " + plot_against + " for mode: " + plot_mode)

    fig = plt.figure(figsize=(16,10))

    mask_pred = fit_data[:,labels=='fMass'].flatten() == mass
    pid_idx = np.where(np.abs(np.array(masses) - mass)<0.001)[0][0]
    output_name = ""
    if(np.sum(mask_pred)!=0):

        y_space = np.linspace(-3., 3., 20*6)
        range_y = [-3.,3.]
        # net_out_tmp = np.array([network(net_x,ort_session=ort_sess_mean).flatten(),network(net_x,ort_session=ort_sess_sigma).flatten()]).T
        # net_y = ((net_out.T[0].flatten()/(fit_data[:,labels=='fTPCSignal'].flatten()*fit_data[:,labels=='fInvDeDxExpTPC'].flatten())[mask_pred])-1.)/(net_out.T[1].flatten())
        if plot_mode == "nsigma":
            if useNN:
                output_name += "NetworkRatioNSigmaBins_"
                net_out_tmp = net_out[mask_pred]
                temp=net_out_tmp.T[1].flatten()-net_out_tmp.T[0].flatten()
                temp[temp==0]=1e-9
                net_y = (((fit_data[:,labels=='fTPCSignal'].flatten()*fit_data[:,labels=='fInvDeDxExpTPC'].flatten())[mask_pred] - net_out_tmp.T[0].flatten()))/(temp)
            else:
                output_name += "BBRatioNSigmaBins_"
                net_y = (fit_data[:,labels=='fTPCSignal'].flatten()[mask_pred]*fit_data[:,labels=='fInvDeDxExpTPC'].flatten()[mask_pred] - 1.)/0.07
        elif plot_mode == "rel_sigma":
            y_space = np.linspace(0.03, 0.2, int((0.2-0.03)/0.001))
            range_y = [0.03,0.2]
            if useNN:
                output_name += "NetworkRelativeSigmaBins_"
                net_out_tmp = net_out[mask_pred]
                mean_correction_factor = net_out_tmp.T[0].flatten()
                net_y = net_out_tmp.T[1].flatten() - mean_correction_factor
            else:
                LOG.info("Relative sigma plot only available for NN mode")
                return
        elif plot_mode == "raw_relsigma_fitted":
            output_name += "RawRelativeSigmaBins_"
            range_y = [-0.2,0.2]
            net_y = (fit_data[:,labels=='fTPCSignal'].flatten()*fit_data[:,labels=='fInvDeDxExpTPC'].flatten() - 1.)[mask_pred]
        else:
            LOG.info("Unknown plot output: " + plot_mode)
            return
        mask_net_out = np.logical_and(net_y>sigma_range[0], net_y<sigma_range[1])

        hist_x_values = transform_x(fit_data[:,labels==plot_against].flatten()[mask_pred])
        if log_x:
            x_space = np.logspace(range_hists[i][0], range_hists[i][1], bins_sig_mean+1)
            bins = np.log10(np.logspace(range_hists[i][0], range_hists[i][1], bins_sig_mean+1))
            raw_x_data = hist_x_values.copy()
            raw_x_data[raw_x_data <= 0] = 1e-9 # Replace 0 values with 1e-9 to keep log defined
            x_data = np.log10(raw_x_data[mask_net_out])
        else:
            x_space = np.linspace(range_hists[i][0], range_hists[i][1], bins_sig_mean+1)
            bins = np.linspace(range_hists[i][0], range_hists[i][1], bins_sig_mean+1)
            x_data = hist_x_values[mask_net_out]

        # mean values for assigned species

        remove_nan = (~np.isnan(x_data)) & (~np.isnan(net_y[mask_net_out])) & (~np.isinf(x_data)) & (~np.isinf(net_y[mask_net_out]))
        x_data = x_data[remove_nan]
        net_y_new = net_y[mask_net_out][remove_nan]
        binned_mean = sc.stats.binned_statistic(x_data, net_y_new,
                            statistic='mean', bins=bins_sig_mean, range=(range_hists[i][0],range_hists[i][1]))
        nogaus_binned_mean = sc.stats.binned_statistic(x_data, net_y_new,
                            statistic='mean', bins=bins_sig_mean, range=(range_hists[i][0],range_hists[i][1]))
        binned_sigma = sc.stats.binned_statistic(x_data, net_y_new,
                            statistic='std', bins=bins_sig_mean, range=(range_hists[i][0],range_hists[i][1]))
        nogaus_binned_sigma = sc.stats.binned_statistic(x_data, net_y_new,
                            statistic='std', bins=bins_sig_mean, range=(range_hists[i][0],range_hists[i][1]))

        # Bin edges in the working (possibly log10) space
        bin_edges = binned_mean[1]
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # centers in log10 space if log_x else linear space

        # Re-fit each bin using the same edges (bins already defined earlier as the working-space edges)
        for j in range(bins_sig_mean):
            try:
                in_bin = (x_data > bin_edges[j]) & (x_data <= bin_edges[j+1])
                if np.any(in_bin):
                    hist1d = np.histogram(net_y_new[in_bin], 200, range=range_y, density=True)
                    if np.sum(hist1d[0]) > 0:
                        bin_centers_y = (hist1d[1][:-1] + hist1d[1][1:]) / 2
                        p0 = [np.max(hist1d[0]), np.mean(net_y_new[in_bin]), np.std(net_y_new[in_bin])]
                        coeff, _ = sc.optimize.curve_fit(
                            gauss, bin_centers_y, hist1d[0],
                            p0=p0, bounds=([0.,-np.inf,0.],[np.inf,np.inf,np.inf])
                        )
                        binned_mean[0][j], binned_sigma[0][j] = coeff[1], coeff[2]
                    else:
                        binned_mean[0][j], binned_sigma[0][j] = np.nan, np.nan
                else:
                    binned_mean[0][j], binned_sigma[0][j] = np.nan, np.nan
            except Exception as e:
                if log_x:
                    LOG.error("Exception at: " + particles[pid_idx] + "/" + plot_against +
                    " for bins: " + str(10**bin_edges[j]) + " " + str(10**bin_edges[j+1]) + " -> " + str(e))
                else:
                    LOG.error("Exception at: " + particles[pid_idx] + "/" + plot_against +
                    " for bins: " + str(bin_edges[j]) + " " + str(bin_edges[j+1]) + " -> " + str(e))
                binned_mean[0][j], binned_sigma[0][j] = np.nan, np.nan
                continue

        # Prepare data for plotting (centers and values)
        y_mean_nogaus = np.array(nogaus_binned_mean[0], dtype=float)
        y_mean = np.array(binned_mean[0], dtype=float)
        y_err_nogaus = np.array(nogaus_binned_sigma[0], dtype=float)
        y_err = np.array(binned_sigma[0], dtype=float)

        if "raw_relsigma" not in plot_mode:
            hist2d_counts, xedges, yedges = np.histogram2d(
                hist_x_values,
                net_y,
                bins=(x_space, y_space)
            )
            positive_counts = hist2d_counts[hist2d_counts > 0]
            color_norm = None
            if positive_counts.size:
                vmin = positive_counts.min()
                vmax = positive_counts.max()
                vmin = max(vmin, 1e-9)
                if vmax <= vmin:
                    vmax = vmin * 1.0001
                color_norm = LogNorm(vmin=vmin, vmax=vmax)
            im = plt.pcolormesh(xedges, yedges, hist2d_counts.T, cmap=jet_map_alpha, norm=color_norm)
            if positive_counts.size:
                plt.colorbar(im, pad=0.01, aspect=25)
            else:
                LOG.warning(f"Skipping LogNorm colorbar for {particles[pid_idx]} vs {plot_against} (no histogram entries)")
            if log_x:
                x_centers_plot = 10**bin_centers
                if plot_mode == "nsigma":
                    plt.errorbar(x_centers_plot, y_mean, yerr=y_err, xerr=0,
                        fmt='.', capsize=2., c='black', ls='none', elinewidth=1.,
                        label="Gauss-fit, mean and sigma")
                    plt.ylabel(r'$\frac{dE/dx_{TPC} - dE/dx_{exp}}{\sigma_{exp}}$ [#]', fontsize=fontsize_axislabels)
                    plt.ylim(-3.,3.)
                elif plot_mode == "rel_sigma":
                    plt.scatter(x_centers_plot, y_mean, c='black', label="Gauss-fit, mean")
                    # plt.scatter(x_centers_plot, y_mean_nogaus, c='blue', label="Bin-mean, mean")
                    plt.axhline(0.07, c='blue', ls='--', lw=2, label="Nominal dE/dx resolution at MIP")
                    plt.ylabel(r'$\sigma_{rel}(dE/dx_{TPC})$', fontsize=fontsize_axislabels)
                    plt.ylim(0.03,0.2)
                plt.xscale('log')
            else:
                x_centers_plot = bin_centers
                if plot_mode == "nsigma":
                    plt.errorbar(x_centers_plot, y_mean, yerr=y_err, xerr=0,
                        fmt='.', capsize=2., c='black', ls='none', elinewidth=1.,
                        label="Gauss-fit, mean and sigma")
                    plt.ylabel(r'$\frac{dE/dx_{TPC} - dE/dx_{exp}}{\sigma_{exp}}$ [#]', fontsize=fontsize_axislabels)
                    plt.ylim(-3.,3.)
                elif plot_mode == "rel_sigma":
                    plt.scatter(x_centers_plot, y_mean, c='black', label="Gauss-fit, mean")
                    plt.scatter(x_centers_plot, y_mean_nogaus, c='blue', label="Bin-mean, mean")
                    plt.axhline(0.07, c='blue', ls='--', lw=2, label="Nominal dE/dx resolution at MIP")
                    plt.ylabel(r'$\sigma_{rel}(dE/dx_{TPC})$', fontsize=fontsize_axislabels)
                    plt.ylim(0.03,0.2)
        else:
            if log_x:
                x_centers_plot = 10**bin_centers
                plt.xscale('log')
            else:
                x_centers_plot = bin_centers
            plt.axhline(0.07, c='blue', ls='--', lw=2, label="Nominal dE/dx resolution at MIP")
            plt.scatter(x_centers_plot, y_err, c='black', label="Gauss-fit, sigma")
            plt.scatter(x_centers_plot, y_err_nogaus, c='blue', label="Bin-sigma, sigma")
            plt.ylim(0.03,0.2)
            plt.xlim(x_space[0], x_space[-1])

        plt.xlabel(xlabel, fontsize=fontsize_axislabels)
        plt.grid()
        plt.legend(title="Species: " + particles[pid_idx])
        plt.savefig(qa_dir + "/" + plot_against + "/" + output_name + particles[pid_idx] + "_" + plot_against + '.pdf', bbox_inches='tight')
        plt.close()

def separation_power(useNN=0, useMassAssumption=0, momentumSelection=[0.3,0.4],
                     gauss_labels = {
                         0: ["electrons", "pions"],
                         2: ["pions", "electrons"]
                     },
                     y_ranges = {
                        0: [-10.,3.],
                        2: [-3.,10.]
                     },
                     initial_params = {
                         0: [None,0.,1.,None,-4.,1.],
                         2: [None,0.,1.,None,4.,1.]
                     }):

    plot_labels = {
        "pions": ["Pions", "red"],
        "electrons": ["Electrons", "orange"]
    }

    os.makedirs(qa_dir + "/SeparationPower", exist_ok=True)

    ### usemMassAssumption is the index in the masses array: 0 = electrons, 2 = pions

    fig = plt.figure(figsize=(16,10))
    y_bins = np.linspace(*(y_ranges[useMassAssumption]),200)

    # input = fit_data.copy()
    selection = (fit_data[:,labels=='fTPCInnerParam'].flatten() > 0.3) * (fit_data[:,labels=='fTPCInnerParam'].flatten() < 0.4)
    # input = input[selection]
    net_out_tmp = net_out[selection]
    fit_data_tmp = fit_data[selection]

    LOG.info("Looking for provided BB params")
    BBparams = CONFIG['output']['fitBBGraph']['BBparameters']
    #Using extension to account for christians syntax
    BBparams.extend([50,2.3])
    LOG.info(f"Found BB params {BBparams}")
    fit_data_tmp[:,labels=='fInvDeDxExpTPC'] = (1./BetheBlochAleph(fit_data_tmp[:,labels=='fTPCInnerParam'].flatten()/masses[useMassAssumption], params = BBparams)).reshape(-1,1)
    LOG.info(f"Used fitted BB params {BBparams}")

    fit_data_tmp[:,labels=='fMass'] = masses[useMassAssumption]
    fit_bounds_double_gauss = ([0.,-10.,0.,0.,-10.,0.],[np.inf,10.,3.,np.inf,10.,3.])

    if useNN:
        # net_out = network(input[:,mask_X],ort_session=ort_sess_full)
        temp=net_out_tmp.T[1].flatten()-net_out_tmp.T[0].flatten()
        temp[temp==0]=1e-9
        output = (((fit_data_tmp[:,labels=='fTPCSignal'].flatten()*fit_data_tmp[:,labels=='fInvDeDxExpTPC'].flatten()) - net_out_tmp.T[0].flatten()))/(temp)
    else:
        output = (fit_data_tmp[:,labels=='fTPCSignal'].flatten()*fit_data_tmp[:,labels=='fInvDeDxExpTPC'].flatten() - 1.)/0.07

    hist1d = np.histogram(output, bins=y_bins, range=(-3.,3.), density=True)
    initial_params[useMassAssumption][0] = np.max(hist1d[0])
    initial_params[useMassAssumption][3] = np.max(hist1d[0])
    popt, pcov = sc.optimize.curve_fit(double_gauss, y_bins[:-1], hist1d[0], p0=initial_params[useMassAssumption], bounds=fit_bounds_double_gauss)
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
    plt.tight_layout()
    if useNN:
        plt.savefig(qa_dir + '/SeparationPower/SeparationPower_' + particles[useMassAssumption] + 'MassHypothesis_NN.pdf', bbox_inches='tight')
    else:
        plt.savefig(qa_dir + '/SeparationPower/SeparationPower_' + particles[useMassAssumption] + 'MassHypothesis_BB.pdf', bbox_inches='tight')
    plt.close()


for param in ['fTPCInnerParam', 'fTgl', 'fNormMultTPC', 'fNormNClustersTPC', 'fFt0Occ']:
    os.system("rm -rf " + qa_dir + "/" + param)
    os.makedirs(qa_dir + "/" + param)

# Create HadronicRate folder only if flag is enabled (old-script compatibility)
if HadronicRateBool:
    os.system("rm -rf " + qa_dir + "/fHadronicRate")
    os.makedirs(qa_dir + "/fHadronicRate")
    LOG.debug("Created folder for QA plots for HadronicRate")
else:
    LOG.debug("No folder for QA plots for HadronicRate (disabled)")

for i, mass in enumerate(np.sort(np.unique(fit_data[:,labels=='fMass'].flatten()))):

    def transform_ncl(x):
        return 152./(x**2)

    val = CONFIG['trainNeuralNetOptions'].get('isSmallSystem', 'False')
    IsSmallSystem = val.lower() == "true"

    QA2D_NSigma_vs_Var(i, mass, plot_against = 'fTPCInnerParam', log_x = True, range_hists = [[-1.,1.]]*6, useNN=False)
    QA2D_NSigma_vs_Var(i, mass, plot_against = 'fTPCInnerParam', log_x = True, range_hists = [[-1.,1.]]*6)
    QA2D_NSigma_vs_Var(i, mass, plot_against = 'fTPCInnerParam', log_x = True, range_hists = [[-1.,1.]]*6, plot_mode="rel_sigma")
    QA2D_NSigma_vs_Var(i, mass, plot_against = 'fTPCInnerParam', log_x = True, range_hists = [[-1.,1.]]*6, plot_mode="raw_relsigma_fitted")

    QA2D_NSigma_vs_Var(i, mass, plot_against = 'fTgl', log_x = False, range_hists = [[-1.,1.]]*6, xlabel = r'tan($\lambda$)')
    QA2D_NSigma_vs_Var(i, mass, plot_against = 'fTgl', log_x = False, range_hists = [[-1.,1.]]*6, useNN=False, xlabel = r'tan($\lambda$)')
    QA2D_NSigma_vs_Var(i, mass, plot_against = 'fTgl', log_x = False, range_hists = [[-1.,1.]]*6, xlabel = r'tan($\lambda$)', plot_mode="rel_sigma")
    QA2D_NSigma_vs_Var(i, mass, plot_against = 'fTgl', log_x = False, range_hists = [[-1.,1.]]*6, xlabel = r'tan($\lambda$)', plot_mode="raw_relsigma_fitted")

    QA2D_NSigma_vs_Var(i, mass, plot_against = 'fNormNClustersTPC', log_x = False, bins_sig_mean = 152, range_hists = [[0.5,152.5]]*6, transform_x = transform_ncl, xlabel = r'NCl')
    QA2D_NSigma_vs_Var(i, mass, plot_against = 'fNormNClustersTPC', log_x = False, bins_sig_mean = 152, range_hists = [[0.5,152.5]]*6, useNN=False, transform_x = transform_ncl, xlabel = r'NCl')
    QA2D_NSigma_vs_Var(i, mass, plot_against = 'fNormNClustersTPC', log_x = False, bins_sig_mean = 152, range_hists = [[0.5,152.5]]*6, transform_x = transform_ncl, xlabel = r'NCl', plot_mode="rel_sigma")
    QA2D_NSigma_vs_Var(i, mass, plot_against = 'fNormNClustersTPC', log_x = False, bins_sig_mean = 152, range_hists = [[0.5,152.5]]*6, transform_x = transform_ncl, xlabel = r'NCl', plot_mode="raw_relsigma_fitted")

    if IsSmallSystem:
        QA2D_NSigma_vs_Var(i, mass, plot_against = 'fNormMultTPC', log_x = True, range_hists = [[-2,0.]]*6, xlabel = r'norm. TPC multiplicity')
        QA2D_NSigma_vs_Var(i, mass, plot_against = 'fNormMultTPC', log_x = True, range_hists = [[-2,0.]]*6, useNN=False, xlabel = r'norm. TPC multiplicity')
        QA2D_NSigma_vs_Var(i, mass, plot_against = 'fNormMultTPC', log_x = True, range_hists = [[-2,0.]]*6, xlabel = r'norm. TPC multiplicity', plot_mode="rel_sigma")
        QA2D_NSigma_vs_Var(i, mass, plot_against = 'fNormMultTPC', log_x = True, range_hists = [[-2,0.]]*6, xlabel = r'norm. TPC multiplicity', plot_mode="raw_relsigma_fitted")
        QA2D_NSigma_vs_Var(i, mass, plot_against = 'fFt0Occ', log_x = True, range_hists = [[-3.,0.]]*6, xlabel = r'norm. FT0 occupancy')
        QA2D_NSigma_vs_Var(i, mass, plot_against = 'fFt0Occ', log_x = True, range_hists = [[-3.,0.]]*6, useNN=False, xlabel = r'norm. FT0 occupancy')
        QA2D_NSigma_vs_Var(i, mass, plot_against = 'fFt0Occ', log_x = True, range_hists = [[-3.,0.]]*6, xlabel = r'norm. FT0 occupancy', plot_mode="rel_sigma")
        QA2D_NSigma_vs_Var(i, mass, plot_against = 'fFt0Occ', log_x = True, range_hists = [[-3.,0.]]*6, xlabel = r'norm. FT0 occupancy', plot_mode="raw_relsigma_fitted")
    else:
        QA2D_NSigma_vs_Var(i, mass, plot_against = 'fNormMultTPC', log_x = False, range_hists = [[0.,3.]]*6, xlabel = r'norm. TPC multiplicity')
        QA2D_NSigma_vs_Var(i, mass, plot_against = 'fNormMultTPC', log_x = False, range_hists = [[0.,3.]]*6, useNN=False, xlabel = r'norm. TPC multiplicity')
        QA2D_NSigma_vs_Var(i, mass, plot_against = 'fNormMultTPC', log_x = False, range_hists = [[0.,3.]]*6, xlabel = r'norm. TPC multiplicity', plot_mode="rel_sigma")
        QA2D_NSigma_vs_Var(i, mass, plot_against = 'fNormMultTPC', log_x = False, range_hists = [[0.,3.]]*6, xlabel = r'norm. TPC multiplicity', plot_mode="raw_relsigma_fitted")
        QA2D_NSigma_vs_Var(i, mass, plot_against = 'fFt0Occ', log_x = True, range_hists = [[-1.,1.]]*6, xlabel = r'norm. FT0 occupancy')
        QA2D_NSigma_vs_Var(i, mass, plot_against = 'fFt0Occ', log_x = True, range_hists = [[-1.,1.]]*6, useNN=False, xlabel = r'norm. FT0 occupancy')
        QA2D_NSigma_vs_Var(i, mass, plot_against = 'fFt0Occ', log_x = True, range_hists = [[-1.,1.]]*6, xlabel = r'norm. FT0 occupancy', plot_mode="rel_sigma")
        QA2D_NSigma_vs_Var(i, mass, plot_against = 'fFt0Occ', log_x = True, range_hists = [[-1.,1.]]*6, xlabel = r'norm. FT0 occupancy', plot_mode="raw_relsigma_fitted")

    if(HadronicRateBool):
        QA2D_NSigma_vs_Var(i, mass, plot_against = 'fHadronicRate', log_x = True, range_hists = [[-1.,0.3]]*6, xlabel = r'Hadronic Rate [50kHz]')
        QA2D_NSigma_vs_Var(i, mass, plot_against = 'fHadronicRate', log_x = True, range_hists = [[-1.,0.3]]*6, useNN=False, xlabel = r'Hadronic Rate [50kHz]')
        QA2D_NSigma_vs_Var(i, mass, plot_against = 'fHadronicRate', log_x = True, range_hists = [[-1.,0.3]]*6, plot_mode="rel_sigma",xlabel = r'Hadronic Rate [50kHz]')
        QA2D_NSigma_vs_Var(i, mass, plot_against = 'fHadronicRate', log_x = True, range_hists = [[-1.,0.3]]*6, plot_mode="raw_relsigma_fitted",xlabel = r'Hadronic Rate [50kHz]')

separation_power(useNN=1, useMassAssumption=0)
separation_power(useNN=0, useMassAssumption=0)
separation_power(useNN=1, useMassAssumption=2)
separation_power(useNN=0, useMassAssumption=2)