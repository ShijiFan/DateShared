"""Feasibility only: WFS resultType=hits counts per frozen class (FR_MAPPING_FROZEN.csv) for candidate AOIs and years.
No geometries downloaded. Usage: python -I count_fr.py -> fr_counts.csv"""
import csv, os, time, urllib.parse, urllib.request, re
HERE = os.path.dirname(os.path.abspath(__file__))
AOI = {'FR-A_PaysDeCaux': (0.55, 49.55, 1.10, 49.90), 'FR-B_FlandreInt': (2.25, 50.55, 2.80, 50.85)}
YEARS = (2019, 2023)
M = {}
for r in csv.DictReader(open(os.path.join(HERE, 'FR_MAPPING_FROZEN.csv'), encoding='utf-8')): M.setdefault(r['class'], []).append(r['code_cultu'])
rows = []
for name, (a, b, c, d) in AOI.items():
    for y in YEARS:
        for cls, codes in list(M.items()) + [('ALL', None)]:
            for amin in (0.0, 0.5):
                f = f"BBOX(geom,{a},{b},{c},{d},'EPSG:4326')"          # CQL BBOX on this server: lon,lat order (checked: 26810 parcels FR-A 2023)
                if codes: f += " AND code_cultu IN (" + ",".join(f"'{x}'" for x in codes) + ")"
                if amin: f += f" AND surf_parc >= {amin}"
                q = urllib.parse.urlencode(dict(SERVICE='WFS', VERSION='2.0.0', REQUEST='GetFeature', TYPENAMES=f'RPG.{y}:parcelles_graphiques', resultType='hits', CQL_FILTER=f))
                txt = urllib.request.urlopen('https://data.geopf.fr/wfs/ows?' + q, timeout=120).read().decode('utf-8', 'replace')
                m = re.search(r'numberMatched="(\d+)"', txt); n = int(m.group(1)) if m else -1
                rows.append(dict(aoi=name, year=y, cls=cls, min_ha=amin, n=n)); print(name, y, cls, amin, n, flush=True); time.sleep(0.3)
with open(os.path.join(HERE, 'fr_counts.csv'), 'w', newline='', encoding='utf-8') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
