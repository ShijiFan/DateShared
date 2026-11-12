"""B-full FROZEN candidates for the confirm split (PROTOCOL §4). Selected on the search split by S (min over directions of
median paired BA difference vs hpj_cov01): k1_decay_cond S=+0.0022; tie k4_net_cond / k2_clsamp_cond S=+0.0020 -> lower
complexity k4 (6 vs 10 global parameters). j2 (toy-B equation as-is, conditioned) is carried as a reference, not selected.
Definitions imported unchanged from cands_r3.py (SHA256 prefix F23932E35C8ABD3A)."""
from cands_r3 import CANDS as R3
CANDS = [c for c in R3 if c['name'] in ('em_cov01', 'hpj_cov01', 'j2_hpj_wc_cond', 'k1_decay_cond', 'k4_net_cond')]
