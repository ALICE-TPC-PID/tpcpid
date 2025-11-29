### General particles in dE/dx distribution

particle_info = {
    "charges": [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        2.0
    ],
    "masses": [
        0.00051099895,
        0.1056583755,
        0.13957039,
        0.493677,
        0.93827208816,
        1.875613115,
        2.8089211,
        2.8083916
    ],
    "particle_labels": [
        "e",
        "$\\mu$",
        "$\\pi$",
        "$K$",
        "$p$",
        "$d$",
        "$t$",
        "$^3$He"
    ],
    "particles": [
        "Electrons",
        "Muons",
        "Pions",
        "Kaons",
        "Protons",
        "Deuteron",
        "Triton",
        "Helium3"
    ]
}

### V0 particles

v0_cut_dict = {
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