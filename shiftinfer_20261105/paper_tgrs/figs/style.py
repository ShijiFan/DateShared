"""Shared figure style for the TGRS manuscript: IEEE print widths, serif text matching IEEEtran (Times), validated categorical palette
(dataviz reference slots 1-4; contrast WARN for slots 3-4 -> always paired with distinct markers / direct labels), lowercase panel tags."""
import matplotlib as mpl
from matplotlib import font_manager
W1, W2 = 3.5, 7.16                                            # IEEE single / double column width (in)
SERIF = 'Times New Roman' if any('Times New Roman' in f.name for f in font_manager.fontManager.ttflist) else 'DejaVu Serif'
COL = {'hpj': '#2a78d6', 'lw': '#eb6834', 'j2': '#1baf7a', 'lowrank': '#eda100', 'em': '#52514e', 'dann': '#6250d6'}
MK = {'hpj': 'o', 'lw': 's', 'j2': '^', 'lowrank': 'D', 'em': 'v', 'dann': 'P'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e6e5e0'


def setup():
    mpl.rcParams.update({'font.family': 'serif', 'font.serif': [SERIF], 'mathtext.fontset': 'stix', 'font.size': 8, 'axes.labelsize': 8,
                         'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                         'axes.linewidth': 0.6, 'lines.linewidth': 1.4, 'savefig.dpi': 600, 'pdf.fonttype': 42, 'ps.fonttype': 42})


def tag(ax, s):
    ax.text(-0.02, 1.02, f'({s})', transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color=INK)
