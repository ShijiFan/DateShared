"""Build refs.bib: copy verified entries from the JAG manuscript bib (read only) + add speech-adaptation entries (bibliographic data from
NOVELTY_P9_BASIS_ZH.md; Lasry & Stern journal name marked for verification before submission)."""
import os, re
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = r'E:\research\SAR\multiyear_20261017\paper\submission_JAG_20261023\LaTeX_source\references.bib'
want = ['HyP3', 'Ledoit2004', 'Pelletier2019', 'Bruzzone2001', 'Ganin2016', 'Li2018adabn', 'Attema1978']
txt = open(SRC, encoding='utf-8').read(); out = []
for k in want:
    m = re.search(r'@\w+\{' + re.escape(k) + r',.*?\n\}', txt, re.S); assert m, k; out.append(m.group(0))
out.append("""@article{LasryStern1984,
 author={Lasry, Moshe J. and Stern, Richard M.},
 title={A posteriori estimation of correlated jointly {G}aussian mean vectors},
 journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
 volume={PAMI-6}, number={4}, pages={530--535}, year={1984}, doi={10.1109/TPAMI.1984.4767559}
}""")
out.append("""@article{Kuhn2000,
 author={Kuhn, Roland and Junqua, Jean-Claude and Nguyen, Patrick and Niedzielski, Nancy},
 title={Rapid speaker adaptation in eigenvoice space},
 journal={IEEE Transactions on Speech and Audio Processing},
 volume={8}, number={6}, pages={695--707}, year={2000}, doi={10.1109/89.876308}
}""")
out.append("""@inproceedings{He2023,
 author={He, Huan and Queen, Owen and Koker, Teddy and Cuevas, Consuelo and Tsiligkaridis, Theodoros and Zitnik, Marinka},
 title={Domain adaptation for time series under feature and label shifts},
 booktitle={Proceedings of the 40th International Conference on Machine Learning},
 series={PMLR}, volume={202}, pages={12746--12774}, year={2023}
}""")
out.append("""@article{Mohammadi2024,
 author={Mohammadi, Sina and Belgiu, Mariana and Stein, Alfred},
 title={A source-free unsupervised domain adaptation method for cross-regional and cross-time crop mapping from satellite image time series},
 journal={Remote Sensing of Environment}, volume={314}, pages={114385}, year={2024}, doi={10.1016/j.rse.2024.114385}
}""")
open(os.path.join(HERE, 'refs.bib'), 'w', encoding='utf-8').write('\n\n'.join(out) + '\n'); print('refs.bib', len(out))
# 2026-11-10: verified additions (plan/REFS_ADD_ZH.md)
with open(os.path.join(HERE, 'refs.bib'), 'a', encoding='utf-8') as f: f.write(chr(10) + open(os.path.join(HERE, 'refs_add.bib'), encoding='utf-8').read())
