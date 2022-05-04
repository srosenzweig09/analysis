import sys

def createDataCard(location, sigROOT, h_name, err_dict, outdir, no_bkg_stats=True, MCstats=True):
    """Creates datacard for a given tree."""
    bkg_crtf = err_dict['bkg_crtf']
    bkg_vrpred = err_dict['bkg_vrpred']
    bkg_vr_normval = err_dict['bkg_vr_normval']

    dataCardText = [
        'imax 1  number of channels\n',
        'jmax 1  number of backgrounds\n',
        'kmax *  number of nuisance parameters\n',
        '-------\n',
        f'shapes sig bin1 {location}/{sigROOT}.root {h_name}\n',
        f'shapes bkg bin1 {location}/data.root h_dat\n',
        f'shapes data_obs bin1 {location}/data.root h_dat\n',
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
        f'error  lnN   -        {bkg_crtf}\n',
        f'sd     lnN   -        {bkg_vrpred}\n',
        f'k      lnN   -        {bkg_vr_normval}\n'
    ]
    if MCstats: dataCardText.append('* autoMCStats 10\n')
    if no_bkg_stats: sigROOT = sigROOT + '_nobkgstats'
    print(f"{outdir}/{sigROOT}.txt")

    with open(f"{outdir}/{sigROOT}.txt", "w") as f:
        f.writelines(dataCardText)
