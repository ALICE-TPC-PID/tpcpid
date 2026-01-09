import matplotlib.pyplot as plt
import numpy as np

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