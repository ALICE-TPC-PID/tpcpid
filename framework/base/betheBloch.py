def BetheBlochAleph(bg, params):
    beta = bg/np.sqrt(1.+ bg*bg)
    aa   = beta**params[3]
    bb   = bg**(-params[4])
    bb   = np.log(params[2]+bb)
    charge_factor = params[5]         # params[5] = mMIP, params[6] = mChargeFactor #usually its the other way around. Here its just for simplicity to copy it directly from the google sheet
    final = (params[1]-aa-bb)*params[0]*charge_factor/aa
    return final