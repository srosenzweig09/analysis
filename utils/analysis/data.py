def createDataCard(tree, location, rootFile):
    """Creates datacard for a given tree."""
    dataCardText = [
        'imax 1  number of channels\n',
        'jmax 1  number of backgrounds\n',
        'kmax *  number of nuisance parameters\n',
        '-------\n',
        f'shapes sig bin1 {location}/{rootFile}.root h_sig\n',
        f'shapes bkg bin1 {location}/{rootFile}.root h_dat\n',
        f'shapes data_obs bin1 {location}/{rootFile}.root h_dat\n',
        '_______\n',
        'bin bin1\n',
        'observation -1\n',
        '-------\n',
        'bin          bin1     bin1\n',
        'process      sig      bkg\n',
        'process      0        1 \n',
        'rate         -1       -1\n',
        '-------\n',
        '\n',
        'lumi   lnN   1.0      -\n',
        'other  lnN   -         1.\n',
        # f'error  lnN   -        tree.bkg_crtf\n',
        # f'sd     lnN   -        tree.bkg_vrpred\n',
        # f'k      lnN   -        tree.bkg_vr_normval\n',
        '* autoMCStats 10\n'
    ]
    return dataCardText

def calcStats(tree):
    """Calculates bkg_CRtf, """