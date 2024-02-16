import numpy as np
import awkward as ak

def make_bins(style):
    if style == 'linspace': return np.linspace
    elif style == 'arange': return np.arange  

def get_deltaM(higgs, center):
    deltaM = np.column_stack(([abs(m - center) for m in higgs]))
    deltaM = deltaM * deltaM
    deltaM = deltaM.sum(axis=1)
    deltaM = np.sqrt(deltaM)
    return deltaM

def get_region_mask(higgs, center, sr_edge, cr_edge):
   deltaM = np.column_stack(([abs(m - val) for m,val in zip(higgs,center)]))
   deltaM = deltaM * deltaM
   deltaM = deltaM.sum(axis=1)
   deltaM = np.sqrt(deltaM)

   sr_mask = deltaM <= sr_edge 
   cr_mask = (deltaM > sr_edge) & (deltaM <= cr_edge) 

   return sr_mask, cr_mask

def get_hs_ls_masks(sr_mask, cr_mask, ls_mask, hs_mask):
   cr_ls_mask = cr_mask & ls_mask
   cr_hs_mask = cr_mask & hs_mask
   sr_ls_mask = sr_mask & ls_mask
   sr_hs_mask = sr_mask & hs_mask

   return cr_ls_mask, cr_hs_mask, sr_ls_mask, sr_hs_mask

class BDTRegion():
    def __init__(self, config, sixb):

        self.definitions = config['definitions']
        self.shape = float(self.definitions['shape'])
        self.ar_center = float(self.definitions['ARcenter'])
        self.sr_edge = float(self.definitions['SRedge'])
        self.cr_edge = float(self.definitions['CRedge'])
        self.vr_edge = float(self.definitions['VRedge'])

        self.avg_btag = config['avg_btag']
        self.avg_btag_threshold = float(self.avg_btag['threshold'])

        self.bdtparams = config['BDT']
        self.nestimators  = int(self.bdt_params['Nestimators'])
        self.learningrate = float(self.bdt_params['learningRate'])
        self.maxdepth     = int(self.bdt_params['maxDepth'])
        self.minleaves    = int(self.bdt_params['minLeaves'])
        self.gbsubsample  = float(self.bdt_params['GBsubsample'])
        self.randomstate  = int(self.bdt_params['randomState'])
        variables = self.bdt_params['variables']
        if isinstance(variables, str): variables = variables.split(", ")
        self.variables = variables

        self.plot_params = config['plot']
        self.plot_style = self.plot_params['style']
        self.nedges = int(self.plot_params['edges'])
        self.steps = int(self.plot_params['steps'])
        self.minMX = int(self.plot_params['minMX'])
        self.maxMX = int(self.plot_params['maxMX'])

        self.mBins = make_bins(self.plot_style)(self.minMX, self.maxMX, self.nedges)
        self.x_mBins = (self.mBins[:-1] + self.mBins[1:])/2

        self.higgs = [sixb.HX.m, sixb.H1.m, sixb.H2.m]
        try: higgs = [m.to_numpy() for m in higgs]
        except: pass

        self.ar_deltaM = get_deltaM(self.higgs, self.ar_center)
        self.asr_mask = self.ar_deltaM <= self.sr_edge
        if sixb._is_signal: self.asr_avgbtag = sixb.btag_avg[sixb.asr_mask]
        self.acr_mask = (self.ar_deltaM > self.sr_edge) & (self.ar_deltaM <= self.cr_edge) # Analysis CR
        if not sixb._is_signal: self.acr_avgbtag = sixb.btag_avg[self.acr_mask]

        self.ls_mask = sixb.btag_avg < self.avg_btag_threshold
        self.hs_mask = sixb.btag_avg >= self.avg_btag_threshold

        self.acr_ls_mask = self.acr_mask & self.ls_mask
        self.acr_hs_mask = self.acr_mask & self.hs_mask
        self.asr_ls_mask = self.asr_mask & self.ls_mask
        self.asr_hs_mask = self.asr_mask & self.hs_mask
        if not sixb._is_signal:
            self.blind_mask = ~self.asr_hs_mask
            self.asr_hs_mask = np.zeros_like(self.asr_mask)
        self.sr_mask = self.asr_hs_mask

        if self.shape == 'concentric': self.init_concentric(self)
        elif self.shape == 'diagonal': self.init_diagonal(self)
        elif self.shape == 'multiple': self.init_multiple(self)
        else: raise ValueError(f"Unknown shape: {self.shape}")

        self.copy_attributes(sixb)

    def init_concentric(self):
        self.vsr_mask = (self.deltaM <= self.cr_edge) &  (self.deltaM > self.sr_edge)
        self.vcr_mask = (self.deltaM > self.cr_edge) & (self.deltaM <= self.vr_edge)

        self.vcr_ls_mask = self.vcr_mask & self.ls_mask
        self.vcr_hs_mask = self.vcr_mask & self.hs_mask
        self.vsr_ls_mask = self.vsr_mask & self.ls_mask
        self.vsr_hs_mask = self.vsr_mask & self.hs_mask

        self.crtf = round(1 + np.sqrt(1/self.acr_ls_mask.sum()+1/self.acr_hs_mask.sum()),2)
        self.vrtf = 1 + np.sqrt(1/self.vcr_ls_mask.sum()+1/self.vcr_hs_mask.sum())

    def init_diagonal(self):
        self.vr_deltaM = get_deltaM(self.higgs, self.vr_center)
        self.vsr_mask = self.vr_deltaM <= self.sr_edge)
        self.vcr_mask = (self.vr_deltaM > self.sr_edge) & (self.vr_deltaM <= self.cr_edge)

        self.vcr_ls_mask = self.vcr_mask & self.ls_mask
        self.vcr_hs_mask = self.vcr_mask & self.hs_mask
        self.vsr_ls_mask = self.vsr_mask & self.ls_mask
        self.vsr_hs_mask = self.vsr_mask & self.hs_mask

    def init_multiple(self):
        deltaS = (self.cr_edge + self.sr_edge) / np.sqrt(2)
        val_centers = []
        val_centers.append((self.ar_center + deltaS, self.ar_center + deltaS, self.ar_center + deltaS))
        val_centers.append((self.ar_center + deltaS, self.ar_center + deltaS, self.ar_center - deltaS))
        val_centers.append((self.ar_center + deltaS, self.ar_center - deltaS, self.ar_center + deltaS))
        val_centers.append((self.ar_center - deltaS, self.ar_center + deltaS, self.ar_center + deltaS))
        val_centers.append((self.ar_center + deltaS, self.ar_center - deltaS, self.ar_center - deltaS))
        val_centers.append((self.ar_center - deltaS, self.ar_center + deltaS, self.ar_center - deltaS))
        val_centers.append((self.ar_center - deltaS, self.ar_center - deltaS, self.ar_center + deltaS))
        val_centers.append((self.ar_center - deltaS, self.ar_center - deltaS, self.ar_center - deltaS))
        
        self.asr_mask, self.acr_mask = get_region_mask(self.higgs, (125,125,125), self.sr_edge, self.cr_edge)

        vsr_masks, vcr_masks = [], []
        self.vsr_mask = np.repeat(False, len(self.asr_mask))
        self.vcr_mask = np.repeat(False, len(self.asr_mask))
        for center in val_centers:
            vsr_mask, vcr_mask = get_region_mask(self.higgs, center, self.sr_edge, self.cr_edge)
            self.vsr_mask = np.logical_or(self.vsr_mask, vsr_mask)
            self.vcr_mask = np.logical_or(self.vcr_mask, vcr_mask)
            vsr_masks.append(vsr_mask)
            vcr_masks.append(vcr_mask)

        assert ak.any(self.vsr_mask), "No validation region found. :("

        self.vcr_ls_masks, self.vcr_hs_masks = [], []
        self.vsr_ls_masks, self.vsr_hs_masks = [], []
        for vsr_mask, vcr_mask in zip(vsr_masks, vcr_masks):
            vcr_ls_mask, vcr_hs_mask, vsr_ls_mask, vsr_hs_mask = get_hs_ls_masks(vsr_mask, vcr_mask, self.ls_mask, self.hs_mask)

            self.vcr_ls_masks.append(vcr_ls_mask)
            self.vcr_hs_masks.append(vcr_hs_mask)
            self.vsr_ls_masks.append(vsr_ls_mask)
            self.vsr_hs_masks.append(vsr_hs_mask)
        
        self.vcr_ls_mask = np.any(self.vcr_ls_masks, axis=0)
        self.vcr_hs_mask = np.any(self.vcr_hs_masks, axis=0)
        self.vsr_ls_mask = np.any(self.vsr_ls_masks, axis=0)
        self.vsr_hs_mask = np.any(self.vsr_hs_masks, axis=0)

    def copy_attributes(self, dest_cls):
        # but not methods
        for attr_name, attr_value in vars(self).items():
            if not callable(attr_value): setattr(dest_cls, attr_name, attr_value)