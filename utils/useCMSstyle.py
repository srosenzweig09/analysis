import matplotlib as mpl

CMS = {
    "font.sans-serif": ["TeX Gyre Heros", "Helvetica", "Arial"],
    # "font.family": "sans-serif",
    "font.family": "serif",
    "font.size": 22,
    "mathtext.fontset": "custom",
    # "mathtext.rm": "TeX Gyre Heros",
    # "mathtext.bf": "TeX Gyre Heros:bold",
    # "mathtext.sf": "TeX Gyre Heros",
    # "mathtext.it": "TeX Gyre Heros:italic",
    # "mathtext.tt": "TeX Gyre Heros",
    # "mathtext.cal": "TeX Gyre Heros",
    "mathtext.default": "regular",
    "figure.figsize": (10.0, 9.0),
    "axes.titlesize": "medium",
    "axes.labelsize": "medium",
    "axes.unicode_minus": False,
    "axes.linewidth": 2,
    "grid.alpha": 0.8,
    "grid.linestyle": ":",
    "line.linewidth": "2",
    "legend.fontsize": "small",
    "legend.handlelength": 1.5,
    "legend.borderpad": 0.5,
    "legend.frameon": False,
    "xaxis.labellocation": "right",
    "xtick.labelsize": "small",
    "xtick.direction": "in",
    "xtick.major.size": 12,
    "xtick.minor.size": 6,
    "xtick.major.pad": 6,
    # "xtick.top": True,
    "xtick.top": False,
    "xtick.major.top": True,
    "xtick.major.bottom": True,
    "xtick.minor.top": True,
    "xtick.minor.bottom": True,
    # "xtick.minor.visible": True,
    "xtick.minor.visible": False,
    "yaxis.labellocation": "top",
    "ytick.labelsize": "small",
    "ytick.direction": "in",
    "ytick.major.size": 12,
    "ytick.minor.size": 6.0,
    "ytick.right": True,
    "ytick.major.left": True,
    "ytick.major.right": True,
    "ytick.minor.left": True,
    "ytick.minor.right": True,
    "ytick.minor.visible": True,
    "savefig.transparent": False,
    "savefig.bbox": 'tight',
}

# Filter extra (labellocation) items if needed
CMS = {k: v for k, v in CMS.items() if k in mpl.rcParams}

CMSTex = {
    **CMS,
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{siunitx},\sisetup{detect-all}, \
                              \usepackage{helvet},\usepackage{sansmath}, \
                              \sansmath",
}

from matplotlib.pyplot import style as plt_style
plt_style.use(CMS)