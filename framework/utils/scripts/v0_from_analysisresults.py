import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
import matplotlib as mpl

import mplhep as hep
plt.style.use(hep.styles.ALICE)
for key in mpl.rcParams.keys():
    if key.startswith('legend.'):
        mpl.rcParams[key] = mpl.rcParamsDefault[key]

parser = argparse.ArgumentParser(description="V0 Selector Configuration")
parser.add_argument("-i", "--input", type=str, default="v0.root", help="Input ROOT file")
parser.add_argument("-o", "--output", type=str, default="AP.pdf", help="Output PDF file")
args = parser.parse_args()

sys.path.append("../../neural_network_class/NeuralNetworkClasses")
from extract_from_root import *

### Example usage
# cd /path/to/scripts/folder
# python3 /lustre/alice/users/csonnab/TPC/tpcpid-github-official/framework/utils/scripts/v0_from_analysisresults.py --input  /lustre/alice/users/lubynets/tpc/outputs/HY/LHC24ar_pass3_QC1_sampling/588467/AnalysisResults_merge_LHC24ar.root --output ./test.pdf

def import_from_AO2D(filename, tree_name="O2tpcskimv0tree", variables=[
    'fTPCSignal',
    'fTPCInnerParam',
    'fEta',
    'fMass',
    'fNormMultTPC',
    'fNormNClustersTPC',
    'fNormNClustersTPCPID',
    'fPidIndex',
    'fQtV0',
    'fCosPAV0',
    'fPtV0',
    'fRadiusV0',
    'fGammaPsiPair',
    'fFt0Occ',
    'fHadronicRate'
]):
    """
    Import arrays from a ROOT file using uproot.

    Behavior:
    - For TTrees whose key name contains `tree_name`, load the requested branch arrays
      from `variables` and concatenate across matching trees.
    - If a variable name refers to a TH1/TH2 (either at file root or inside a directory
      whose name contains `tree_name`), read its contents via `.to_numpy()` and append
      them. TH1 is stored as (values, xedges); TH2 as (values, xedges, yedges).

    Returns:
    - variables: the original list of variable names/paths
    - np.array(vars): a numpy object array containing the collected items
      (branch arrays and/or histogram tuples). Shapes may differ, so dtype=object.
    """
    f = uproot.open(filename)

    # Prepare storage for all requested variables
    vars = [None] * len(variables)

    # Normalize uproot.to_numpy() return to a consistent tuple we keep:
    # - TH2: (values, xedges, yedges)
    # - TH1: (values, xedges)
    def _normalize_numpy_ret(ret):
        if isinstance(ret, (tuple, list)):
            if len(ret) == 3:  # TH2: (values, xedges, yedges)
                values, xedges, yedges = ret
                return (np.asarray(values), np.asarray(xedges), np.asarray(yedges))
            if len(ret) == 2:  # TH1: (values, xedges) OR TH2 alt: (values, (xedges, yedges))
                values, edges = ret
                if isinstance(edges, (tuple, list)) and len(edges) == 2:  # TH2 alt format
                    xedges, yedges = edges
                    return (np.asarray(values), np.asarray(xedges), np.asarray(yedges))
                # Otherwise treat as TH1
                return (np.asarray(values), np.asarray(edges))
        return None

    # Merge two histogram tuples if edges are identical; else return a list
    def _merge_hist(existing, new):
        # both TH2 (3-tuple)
        if isinstance(existing, (tuple, list)) and len(existing) == 3 and isinstance(new, (tuple, list)) and len(new) == 3:
            v1, xe1, ye1 = existing
            v2, xe2, ye2 = new
            if np.array_equal(xe1, xe2) and np.array_equal(ye1, ye2):
                return (v1 + v2, xe1, ye1)
            return [existing, new]
        # both TH1 (2-tuple)
        if isinstance(existing, (tuple, list)) and len(existing) == 2 and isinstance(new, (tuple, list)) and len(new) == 2:
            v1, e1 = existing
            v2, e2 = new
            if np.array_equal(e1, e2):
                return (v1 + v2, e1)
            return [existing, new]
        # if existing is already a list, append
        if isinstance(existing, list):
            existing.append(new)
            return existing
        # fallback to list of both
        return [existing, new]

    # 1) Iterate over file keys that match `tree_name` and try to read either branches or histograms in that scope
    count = 0
    for k in tqdm(f.keys()):
        if tree_name in k:
            node = f[k]
            for j, v in enumerate(variables):
                # Try as a TTree branch
                arr = None
                try:
                    obj = node[v]
                    if hasattr(obj, "array"):
                        arr = obj.array(library="np")
                except Exception:
                    arr = None

                if arr is not None:
                    if count == 0:
                        vars[j] = arr
                    else:
                        if vars[j] is None:
                            vars[j] = arr
                        else:
                            vars[j] = np.concatenate((vars[j], arr), axis=0)
                    continue

                # Try as a TH1/TH2 in this node
                ret = None
                try:
                    obj = node[v]
                    if hasattr(obj, "to_numpy"):
                        ret = obj.to_numpy()
                except Exception:
                    ret = None

                if ret is not None:
                    norm = _normalize_numpy_ret(ret)
                    if norm is not None:
                        if vars[j] is None:
                            vars[j] = norm
                        else:
                            vars[j] = _merge_hist(vars[j], norm)
            count += 1

    # 2) For any variable still None, try it as a histogram at file root or under directories matching `tree_name`
    for j, v in enumerate(variables):
        if vars[j] is not None:
            continue
        ret = None
        # Direct path (may be "dir/hist" or root-level key)
        try:
            obj = f[v]
            if hasattr(obj, "to_numpy"):
                ret = obj.to_numpy()
        except Exception:
            ret = None

        # If not found at root, search inside nodes that match `tree_name`
        if ret is None:
            for k in f.keys():
                if tree_name in k:
                    try:
                        obj = f[k][v]
                        if hasattr(obj, "to_numpy"):
                            ret = obj.to_numpy()
                            break
                    except Exception:
                        continue

        if ret is not None:
            norm = _normalize_numpy_ret(ret)
            if norm is not None:
                vars[j] = norm

    # Keep arrays/tuples as-is (no list conversion) to preserve edges for plotting
    return variables, np.array(vars, dtype=object)

def plot_th2_from_uproot(h2_like, cmap="viridis", figsize=(10,8),
                         xscale=None, yscale=None, xlabel="X", ylabel="Y", xlimits=None, ylimits=None,
                         title=None,
                         use_lognorm=True, ax=None, subplot_index=None,
                         xedges=None, yedges=None, add_gauss_in_xbins=False, show=True,
                         colorbar_range=None, plot_colorbar=False, plot_legend=False, legend_args=None):
    """
    Plot a TH2-like histogram from diverse input forms:
    - TH2: (values, xedges, yedges) or (values, (xedges, yedges))
    - TH1: (values, xedges) will be plotted as a 1D step/bar using true edges
    - A list of the above: sums values when edges match; otherwise uses the first
    - A plain 2D values array is NOT accepted unless xedges and yedges are provided

    Returns (fig_or_None, ax, artist_or_mesh).
    """
    # Detect TH1: (values, xedges) where xedges is 1D array-like
    if isinstance(h2_like, (list, tuple)) and len(h2_like) == 2:
        values, edges = h2_like
        # TH1 case: edges is a 1D array (not a pair of edges)
        if not (isinstance(edges, (list, tuple)) and len(edges) == 2):
            values = np.asarray(values)
            xed = np.asarray(edges)
            # Plot 1D histogram using true edges
            created_fig = False
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
                created_fig = True
            else:
                fig = ax.figure

            widths = np.diff(xed)
            ax.bar(xed[:-1], values, width=widths, align='edge', alpha=0.8, color='C0')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel if ylabel is not None else "Counts")
            if title is not None:
                ax.set_title(title)
            plt.tight_layout()
            if created_fig:
                plt.show()
            return (fig if created_fig else None), ax, None

    def _normalize_th2_input(obj):
        # Handle numpy object arrays of length 3 or 2
        if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == (3,):
            values, xe, ye = obj.tolist()
            return np.asarray(values), np.asarray(xe), np.asarray(ye)

        if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == (2,):
            values, edges2 = obj.tolist()
            if isinstance(edges2, (list, tuple, np.ndarray)) and len(edges2) == 2:
                xe, ye = edges2
                return np.asarray(values), np.asarray(xe), np.asarray(ye)

        # existing checks...
        if isinstance(obj, (list, tuple)) and len(obj) == 3:
            values, xe, ye = obj
            return np.asarray(values), np.asarray(xe), np.asarray(ye)

        if isinstance(obj, (list, tuple)) and len(obj) == 2:
            values, edges2 = obj
            if isinstance(edges2, (list, tuple)) and len(edges2) == 2:
                xe, ye = edges2
                return np.asarray(values), np.asarray(xe), np.asarray(ye)

        arr = np.asarray(obj)
        if arr.ndim == 2:
            if xedges is None or yedges is None:
                raise RuntimeError("2D values array provided without edges. Pass xedges and yedges to plot true axis values.")
            xe = np.asarray(xedges)
            ye = np.asarray(yedges)
            if xe.size != arr.shape[0] + 1 or ye.size != arr.shape[1] + 1:
                raise RuntimeError(f"Edges length mismatch: expected ({arr.shape[0]}+1, {arr.shape[1]}+1), got ({xe.size}, {ye.size}).")
            return arr, xe, ye

        raise RuntimeError("Unsupported input. Provide TH2 with edges or a 2D array plus xedges/yedges.")


    def add_gaussian_fits_in_xbins(ax, values, xedges, yedges):
        from scipy.optimize import curve_fit

        def gaussian(x, amp, mean, sigma):
            return amp * np.exp(-0.5 * ((x - mean) / sigma) ** 2)

        xcenters = 0.5 * (xedges[:-1] + xedges[1:])
        ycenters = 0.5 * (yedges[:-1] + yedges[1:])

        means = []
        sigmas = []
        valid_x = []

        for i in range(values.shape[0]):  # loop over x-bins
            hist = values[i, :]
            if np.sum(hist) < 10:
                continue
            try:
                # initial guesses
                p0 = [np.max(hist), ycenters[np.argmax(hist)], 0.2]
                popt, _ = curve_fit(gaussian, ycenters, hist, p0=p0)

                amp, mu, sigma = popt
                means.append(mu)
                sigmas.append(sigma)
                valid_x.append(xcenters[i])
            except RuntimeError:
                continue

        # plot results
        if valid_x:
            # ax.errorbar(valid_x, means, yerr=sigmas, fmt='o',
            #             color='white', ecolor='black', elinewidth=1.2,
            #             capsize=2, markersize=5, label="Gaussian fits")
            ax.scatter(valid_x, means, s=20, color='black', label="Gaussian fits, mean")
            ax.scatter(valid_x, sigmas, s=20, color='blue', label="Gaussian fits, sigma")

        return valid_x, means, sigmas


    # Normalize as TH2
    values, xe, ye = _normalize_th2_input(h2_like)

    # prepare normalization (robust with zeros/empty)
    vmax = float(np.max(values)) if values.size else 1.0
    norm = None
    if use_lognorm:
        pos = values > 0
        if np.any(pos):
            vmin = float(np.min(values[pos]))
            if colorbar_range:
                vmin, vmax = colorbar_range
            if vmin > 0 and vmax > 0:
                norm = colors.LogNorm(vmin=vmin, vmax=vmax)

    # plotting
    created_fig = False
    if ax is None:
        if subplot_index is None:
            subplot_index = 1
        fig, ax = plt.subplots(subplot_index, figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    mesh = ax.pcolormesh(xe, ye, values.T, cmap=cmap, norm=norm, shading="auto")

    if plot_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(mesh, cax=cax)
        cbar.set_label("Counts")

    if add_gauss_in_xbins:
        add_gaussian_fits_in_xbins(ax, values, xe, ye)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_yscale(yscale)

    if xlimits is not None:
        ax.set_xlim(xlimits)
    if ylimits is not None:
        ax.set_ylim(ylimits)

    if plot_legend:
        if legend_args:
            ax.legend(**legend_args)
        else:
            ax.legend()

    if created_fig:
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.close(fig)

    return (fig if created_fig else None), ax, mesh

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
        q[~((cutQTK0SLow < q) * (q < cutQTK0SHigh) * (np.abs(alpha) < 0.85))] = np.nan
        return q
    plt.plot(alpha, K0S_CUT(alpha), label="$K^0_S$ selection", color="black", linewidth = 4)

    def K0S_CUT_UPPER(alpha):
        q =  cutQTK0SHigh * np.sqrt(np.abs(1. - alpha**2 / (cutAPK0SHighTop**2)))
        q[~((cutQTK0SLow < q) * (q < cutAPK0SHighTop))] = np.nan
        return q
    plt.plot(alpha, K0S_CUT_UPPER(alpha), color="black", linewidth = 4)

    # Lambda cut
    def LAMBDA_CUT(alpha):
        q = cutAPL1 * np.sqrt(np.abs(1 - ((alpha + cutAPL2)**2) / (cutAPL3**2))) * (cutAlphaLLow < alpha)
        q[~((alpha < cutAlphaLHigh) * (cutQTL < q))] = np.nan
        return q
    plt.plot(alpha, LAMBDA_CUT(alpha), label="$\Lambda + \overline{\Lambda}$ selection", color="blue", linewidth = 4)

    def LAMBDA_CUT_LOW(alpha):
        q = cutAPL1Low * np.sqrt(np.abs(1 - ((alpha + cutAPL2Low)**2) / (cutAPL3Low**2))) * (cutAlphaLLow2 < alpha)
        q[~((alpha < cutAlphaLHigh) * (cutQTL < q))] = np.nan
        return q
    plt.plot(alpha, LAMBDA_CUT_LOW(alpha), color="blue", linewidth = 4)

    # Anti-Lambda cut
    def ANTILAMBDA_CUT(alpha):
        q = cutAPL1 * np.sqrt(np.abs(1 - ((alpha - cutAPL2)**2 / (cutAPL3**2)))) * (alpha < -cutAlphaLLow)
        q[~((-cutAlphaLHigh < alpha) * (cutQTL < q))] = np.nan
        return q
    plt.plot(alpha, ANTILAMBDA_CUT(alpha), color="blue", linewidth = 4)

    def ANTILAMBDA_CUT_LOW(alpha):
        q = cutAPL1Low * np.sqrt(np.abs(1 - ((alpha - cutAPL2Low)**2 / (cutAPL3Low**2)))) * (alpha < -cutAlphaLLow2)
        q[~((-cutAlphaLHigh < alpha) * (cutQTL < q))] = np.nan
        return q
    plt.plot(alpha, ANTILAMBDA_CUT_LOW(alpha), color="blue", linewidth = 4)

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

    ### Adding horizontal and vertical lines

    # plt.plot(alpha, GAMMA_CUT1(alpha), label="Gamma Cut 1", color="purple", linestyle="--", linewidth=4)
    # plt.plot(alpha, GAMMA_CUT2(alpha), label="Gamma Cut 2", color="orange", linestyle="--", linewidth=4)
    plt.plot(alpha, GAMMA_CUT_REGION(alpha), label="$\gamma$ selection", color="purple", linestyle="-", linewidth=4)
    plt.plot([cutAlphaGHigh, cutAlphaGHigh], [cutQTG, 0], color="purple", linewidth=4)
    plt.plot([-cutAlphaGHigh, -cutAlphaGHigh], [cutQTG, 0], color="purple", linewidth=4)

    qT_Lambda = [cutAPL1Low * np.sqrt(np.abs(1 - ((cutAlphaLHigh + cutAPL2Low)**2) / (cutAPL3Low**2))) * (cutAlphaLLow2 < cutAlphaLHigh), cutAPL1 * np.sqrt(np.abs(1 - ((cutAlphaLHigh + cutAPL2)**2) / (cutAPL3**2))) * (cutAlphaLLow < cutAlphaLHigh)]
    plt.plot([cutAlphaLHigh, cutAlphaLHigh], qT_Lambda, color='blue', linewidth=4)
    plt.plot([-cutAlphaLHigh, -cutAlphaLHigh], qT_Lambda, color='blue', linewidth=4)

    plt.plot([-cutAPL2Low-cutAPL3Low, cutAlphaLLow], [cutQTL, cutQTL], color='blue', linewidth=4)
    plt.plot([cutAPL2Low+cutAPL3Low, -cutAlphaLLow], [cutQTL, cutQTL], color='blue', linewidth=4)

    qT_Lambda_2 = cutAPL1 * np.sqrt(np.abs(1 - ((cutAlphaLLow + cutAPL2)**2) / (cutAPL3**2)))
    plt.plot([cutAlphaLLow, cutAlphaLLow], [cutQTL, qT_Lambda_2], color='blue', linewidth=4)
    plt.plot([-cutAlphaLLow, -cutAlphaLLow], [cutQTL, qT_Lambda_2], color='blue', linewidth=4)

    alpha_qK0_high = np.sqrt((1. - (cutQTK0SLow/cutQTK0SHigh)**2)*(cutAPK0SHighTop**2))
    alpha_qK0_low = np.sqrt((1. - (cutQTK0SLow/cutAPK0SLow)**2)*(cutAPK0SHigh**2))
    plt.plot([alpha_qK0_low, alpha_qK0_high], [cutQTK0SLow, cutQTK0SLow], color='black', linewidth=4)
    plt.plot([-alpha_qK0_high, -alpha_qK0_low], [cutQTK0SLow, cutQTK0SLow], color='black', linewidth=4)

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

histograms_of_interest = [
    "hMassK0S",
    "hMassLambda",
    "hMassAntiLambda",
    "hMassOmega",
    "hMassAntiOmega",
    "hV0CosPA",
    "hDCAxyPosToPV",
    "hDCAxyNegToPV",
    "hDCAxyPosToPV",
    "hDCAzPosToPV",
    "hDCAzNegToPV",
    "hDCAV0Dau",
    "hV0APplot",
    "hV0APplotSelected",
    "hV0Psi"
]

def idxvar(var, variables=histograms_of_interest):
    return variables.index(var) if var in variables else None

if not os.path.exists(args.output):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

v0histos_lbl, v0histos_gpucf = import_from_AO2D(args.input, tree_name="v0-selector", variables=histograms_of_interest)

fig, ax = plt.subplots(figsize=(16,12))
plot_cuts(**cut_dict)
xlbl = r"$\alpha = \frac{\mathit{p}_{\parallel}^+ - \mathit{p}_{\parallel}^-}{\mathit{p}_{\parallel}^+ + \mathit{p}_{\parallel}^-}$"
ylbl = r"$\mathit{q}_T$ (GeV/$\mathit{c}$)"
plt.text(0,0.02, horizontalalignment='center', verticalalignment='center', fontsize=35, s=r"$\gamma$", c="black", zorder=2)
plt.text(0,0.185, horizontalalignment='center', verticalalignment='center', fontsize=30, s=r"K$^{S}_0$", c="black", zorder=2)
plt.text(-0.7,0.07, horizontalalignment='center', verticalalignment='center', fontsize=35, s=r"$\overline{\Lambda}$", c="black", zorder=2)
plt.text(0.7,0.07, horizontalalignment='center', verticalalignment='center', fontsize=35, s=r"$\Lambda$", c="black", zorder=2)
plt.text(-0.69, 0.2265, "ALICE performance, 2024\nPb$-$Pb, $\sqrt{s_{NN}}=$5.36 TeV", ha="center", fontsize=20, bbox=dict(facecolor="white", edgecolor="none", boxstyle="square,pad=0.5"))
v0 = plot_th2_from_uproot(v0histos_gpucf[idxvar("hV0APplot", v0histos_lbl)], cmap=plt.cm.jet, ax=ax,
                          xlabel=xlbl, ylabel=ylbl, xscale=None, yscale=None,
                          title=None, show=False, plot_colorbar=True, plot_legend=True,
                          legend_args={"loc": "upper right", "fontsize": 20, "title_fontsize": 20, "framealpha": 1, "title": "V0 selections", "bbox_to_anchor": (1.0, 1.0)})
plt.savefig(args.output, bbox_inches='tight')
plt.show()