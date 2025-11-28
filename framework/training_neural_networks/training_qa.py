"""
File: training_qa.py
Author: Christian Sonnabend
Email: christian.sonnabend@cern.ch
Date: 15/03/2024
"""

import sys, os
import argparse
import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LogNorm
import scipy as sc
import json
import torch

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="configuration.json", help="Path to the configuration file")
args = parser.parse_args()

########### Import the Neural Network class ###########

with open(args.config, 'r') as config_file:
    CONFIG = json.load(config_file)
sys.path.append(os.path.join(CONFIG["output"]["general"]["base_folder"], "framework"))
from base import *
from neural_network_class.NeuralNetworkClasses import *
configurations = import_from_path(CONFIG["trainNeuralNetOptions"]["configuration"])

## Loading data and models

particles               = particle_info["particles"]
particle_labels         = particle_info["particle_labels"]
masses                  = particle_info["masses"]
charges                 = particle_info["charges"]
dict_particles_masses   = dict(zip(particles, masses))
output_folder           = CONFIG["output"]["general"]["training"] #general output dir for training
output_dir              = CONFIG["output"]["trainNeuralNet"]["QApath"] #output directory for QA plots
data_path               = CONFIG["output"]["createTrainingDataset"]["training_data"]

LABELS_X = CONFIG["createTrainingDatasetOptions"]["labels_x"]
LABELS_Y = CONFIG["createTrainingDatasetOptions"]["labels_y"]

### General

jet_map = cm.jet(np.arange(cm.jet.N))
# Set alpha
jet_map[:,-1] = np.linspace(0, 1, cm.jet.N)
jet_map_alpha = ListedColormap(jet_map)

fontsize_axislabels = 17
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

ort_sess_mean = ort.InferenceSession(output_folder+"/networks/network_mean/net_onnx_mean.onnx", sess_options)
ort_sess_sigma = ort.InferenceSession(output_folder+"/networks/network_sigma/net_onnx_sigma.onnx", sess_options)
ort_sess_full = ort.InferenceSession(output_folder+"/networks/network_full/net_onnx_full.onnx", sess_options)

def network(data, ort_session=ort_sess_full):
    return np.array(ort_session.run(None, {'input': (torch.tensor(data).float()).numpy()}))


def QA2D_NSigma_vs_Var(i, mass, plot_against = 'fTPCInnerParam', log_x = True, range_hists = [[-1.,1.]]*6, fitted_particles=['Electrons', 'Pions', 'Kaons', 'Protons', 'Deuterons', 'Tritons'], bins_sig_mean=100, sigma_range = [-3.,3.], useNN = True, xlabel = r'p [GeV/c]', transform_x = lambda x: x):

    print("Creating histograms for", fitted_particles[i], "against", plot_against)

    fig = plt.figure(figsize=(16,9))

    mask_pred = fit_data[:,labels=='fMass'].flatten() == mass
    pid_idx = np.where(np.abs(np.array(masses) - mass)<0.001)[0][0]
    if(np.sum(mask_pred)!=0):

        y_space = np.linspace(-3., 3., 20*6)
        net_x = X[mask_pred]
        # net_out = np.array([network(net_x,ort_session=ort_sess_mean).flatten(),network(net_x,ort_session=ort_sess_sigma).flatten()]).T
        # net_y = ((net_out.T[0].flatten()/(fit_data[:,labels=='fTPCSignal'].flatten()*fit_data[:,labels=='fInvDeDxExpTPC'].flatten())[mask_pred])-1.)/(net_out.T[1].flatten())
        if useNN:
            net_out = network(net_x,ort_session=ort_sess_full)
            temp=net_out.T[1].flatten()-net_out.T[0].flatten()
            temp[temp==0]=1e-9
            net_y = (((fit_data[:,labels=='fTPCSignal'].flatten()*fit_data[:,labels=='fInvDeDxExpTPC'].flatten())[mask_pred] - net_out.T[0].flatten()))/(temp)
        else:
            net_y = (fit_data[:,labels=='fTPCSignal'].flatten()[mask_pred]*fit_data[:,labels=='fInvDeDxExpTPC'].flatten()[mask_pred] - 1.)/0.07
        mask_net_out = np.logical_and(net_y>sigma_range[0], net_y<sigma_range[1])

        if log_x:
            x_space = np.logspace(range_hists[i][0], range_hists[i][1], 20*8)
            bins = np.log10(np.logspace(range_hists[i][0], range_hists[i][1], bins_sig_mean+1))
            # x_data = np.log10(transform_x(fit_data[:,labels==plot_against].flatten()[mask_pred]))[mask_net_out]
            # Extract raw x_data and replace 0 values with a very small number before applying log transformation
            raw_x_data = transform_x(fit_data[:,labels==plot_against].flatten()[mask_pred])
            raw_x_data[raw_x_data <= 0] = 1e-9 # Replace 0 values with 1e-9 x_data
            x_data = np.log10(raw_x_data[mask_net_out])
        else:
            x_space = np.linspace(range_hists[i][0], range_hists[i][1], 20*8)
            bins = np.linspace(range_hists[i][0], range_hists[i][1], bins_sig_mean+1)
            x_data = transform_x(fit_data[:,labels==plot_against].flatten()[mask_pred])[mask_net_out]

        plt.hist2d(transform_x(fit_data[:,labels==plot_against].flatten()[mask_pred]), net_y, bins=(x_space, y_space), range=[range_hists[i],[-3.,3.]], cmap=jet_map_alpha, norm=LogNorm())

        # mean values for assigned species

        remove_nan = (~np.isnan(x_data)) & (~np.isnan(net_y[mask_net_out])) & (~np.isinf(x_data)) & (~np.isinf(net_y[mask_net_out]))
        x_data = x_data[remove_nan]
        net_y = net_y[mask_net_out][remove_nan]
        binned_mean = sc.stats.binned_statistic(x_data, net_y,
                                                statistic='mean', bins=bins_sig_mean, range=(range_hists[i][0],range_hists[i][1]))
        binned_sigma = sc.stats.binned_statistic(x_data, net_y,
                                                statistic='std', bins=bins_sig_mean, range=(range_hists[i][0],range_hists[i][1]))

        for j in range(bins_sig_mean):
            try:
                hist1d = np.histogram(net_y[(x_data>bins[j]) * (x_data<bins[j+1])], bins_sig_mean, range=(-3.,3.), density=True)
                if np.sum(hist1d[0]) > 0:
                    bin_centers = (hist1d[1][:-1] + hist1d[1][1:])/2
                    p0 = [1.,0.,1.]
                    coeff, var_matrix = sc.optimize.curve_fit(gauss, bin_centers, hist1d[0], p0=p0, bounds=([0.,-np.inf,0.],[np.inf,np.inf,np.inf]))
                    binned_mean[0][j], binned_sigma[0][j] = coeff[1], coeff[2]
                else:
                    binned_mean[0][j], binned_sigma[0][j] = None, None
            except Exception as e:
                if log_x:
                    print("Exception at: ", particles[pid_idx], "/", plot_against, "for bins:", 10**bins[j], 10**bins[j+1])
                else:
                    print("Exception at: ", particles[pid_idx], "/", plot_against, "for bins:", bins[j], bins[j+1])
                print(e)
                continue

        if log_x:
            plt.errorbar(10**np.linspace(range_hists[i][0],range_hists[i][1], bins_sig_mean), binned_mean[0], yerr=binned_sigma[0], xerr=0,
                                    fmt='.', capsize=2., c='black', ls='none', elinewidth=1., label="Binned mean and sigma")
            plt.xscale('log')
        else:
            plt.errorbar(np.linspace(range_hists[i][0],range_hists[i][1], bins_sig_mean), binned_mean[0], yerr=binned_sigma[0], xerr=0,
                                    fmt='.', capsize=2., c='black', ls='none', elinewidth=1., label="Binned mean and sigma")

        plt.xlabel(xlabel, fontsize=fontsize_axislabels)
        plt.ylabel(r'$\frac{dE/dx_{TPC} - dE/dx_{exp}}{\sigma_{exp}}$ [σ]', fontsize=fontsize_axislabels)
        # plt.ylim(-5.,5.)
        plt.ylim(-3.,3.)
        plt.grid()
        plt.colorbar(pad=0.01, aspect=25)
        plt.legend(title="Species: " + particles[pid_idx])
        if useNN:
            plt.savefig(output_dir + "/" + plot_against + '/NetworkRatioNSigmaBins_' + particles[pid_idx] + "_" + plot_against + '.pdf', bbox_inches='tight')
        else:
            plt.savefig(output_dir + "/" + plot_against + '/BBRatioNSigmaBins_' + particles[pid_idx] + "_" + plot_against + '.pdf', bbox_inches='tight')
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

    os.makedirs(output_dir + "/SeparationPower", exist_ok=True)

    ### usemMassAssumption is the index in the masses array: 0 = electrons, 2 = pions

    def separation_power(mu1, mu2, sigma1, sigma2):
        return 2*np.abs(mu1-mu2)/(sigma1+sigma2)

    fig = plt.figure(figsize=(16,9))
    y_bins = np.linspace(*(y_ranges[useMassAssumption]),200)

    input = fit_data.copy()
    selection = (input[:,labels=='fTPCInnerParam'].flatten() > 0.3) * (input[:,labels=='fTPCInnerParam'].flatten() < 0.4)
    input = input[selection]
    print("Looking for provided BB params")
    BBparams = CONFIG["output"]["fitBBGraph"]["BBparameters"]
    #Using extension to account for christians syntax
    BBparams.extend([50,2.3])
    print(f"Found BB params {BBparams}")
    input[:,labels=='fInvDeDxExpTPC'] = (1./BetheBlochAleph(fit_data[selection,labels=='fTPCInnerParam'].flatten()/masses[useMassAssumption], params = BBparams)).reshape(-1,1)
    print(f"Used fitted BB params {BBparams}")

    input[:,labels=='fMass'] = masses[useMassAssumption]
    fit_bounds_double_gauss = ([0.,-10.,0.,0.,-10.,0.],[np.inf,10.,3.,np.inf,10.,3.])

    if useNN:
        net_out = network(input[:,mask_X],ort_session=ort_sess_full)
        temp=net_out.T[1].flatten()-net_out.T[0].flatten()
        temp[temp==0]=1e-9
        output = (((input[:,labels=='fTPCSignal'].flatten()*input[:,labels=='fInvDeDxExpTPC'].flatten()) - net_out.T[0].flatten()))/(temp)
    else:
        output = (input[:,labels=='fTPCSignal'].flatten()*input[:,labels=='fInvDeDxExpTPC'].flatten() - 1.)/0.07

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
    if useNN:
        plt.savefig(output_dir + '/SeparationPower/SeparationPower_' + particles[useMassAssumption] + 'MassHypothesis_NN.pdf', bbox_inches='tight')
    else:
        plt.savefig(output_dir + '/SeparationPower/SeparationPower_' + particles[useMassAssumption] + 'MassHypothesis_BB.pdf', bbox_inches='tight')
    plt.close()


for param in ['fTPCInnerParam', 'fTgl', 'fNormMultTPC', 'fNormNClustersTPC', 'fFt0Occ']:
    os.system("rm -rf " + output_dir + "/" + param)
    os.makedirs(output_dir + "/" + param)

for i, mass in enumerate(np.sort(np.unique(fit_data[:,labels=='fMass'].flatten()))):

    def transform_ncl(x):
        return 152./(x**2)

    QA2D_NSigma_vs_Var(i, mass, plot_against = 'fTPCInnerParam', log_x = True, range_hists = [[-1.,1.]]*6, useNN=False)
    QA2D_NSigma_vs_Var(i, mass, plot_against = 'fTPCInnerParam', log_x = True, range_hists = [[-1.,1.]]*6)
    QA2D_NSigma_vs_Var(i, mass, plot_against = 'fTgl', log_x = False, range_hists = [[-1.,1.]]*6, xlabel = r'tan($\lambda$)')
    QA2D_NSigma_vs_Var(i, mass, plot_against = 'fTgl', log_x = False, range_hists = [[-1.,1.]]*6, useNN=False, xlabel = r'tan($\lambda$)')
    QA2D_NSigma_vs_Var(i, mass, plot_against = 'fNormMultTPC', log_x = False, range_hists = [[0.,3.]]*6, xlabel = r'norm. multiplicity')
    QA2D_NSigma_vs_Var(i, mass, plot_against = 'fNormMultTPC', log_x = False, range_hists = [[0.,3.]]*6, useNN=False, xlabel = r'norm. multiplicity')
    QA2D_NSigma_vs_Var(i, mass, plot_against = 'fNormNClustersTPC', log_x = False, bins_sig_mean = 152, range_hists = [[1.,152.]]*6, transform_x = transform_ncl, xlabel = r'NCl')
    QA2D_NSigma_vs_Var(i, mass, plot_against = 'fNormNClustersTPC', log_x = False, bins_sig_mean = 152, range_hists = [[1.,152.]]*6, useNN=False, transform_x = transform_ncl, xlabel = r'NCl')
    QA2D_NSigma_vs_Var(i, mass, plot_against = 'fFt0Occ', log_x = True, range_hists = [[-1.,1.]]*6, xlabel = r'norm. FT0 Occupancy')
    QA2D_NSigma_vs_Var(i, mass, plot_against = 'fFt0Occ', log_x = True, range_hists = [[-1.,1.]]*6, useNN=False, xlabel = r'norm. FT0 Occupancy')

# separation_power(useNN=1, useMassAssumption=0)
# separation_power(useNN=0, useMassAssumption=0)
# separation_power(useNN=1, useMassAssumption=2)
# separation_power(useNN=0, useMassAssumption=2)

print("[CRITICAL]: Just the old plots are created. Please implement functionality of October version")