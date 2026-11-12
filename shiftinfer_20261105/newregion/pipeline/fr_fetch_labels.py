"""Fetch IGN RPG parcelles_graphiques for the FR-A bbox via WFS (paged), keep code_cultu/id_parcel/surf_parc + geometry.
Licence Ouverte 2.0. Output (data dir, separate from code): D:\\sar_data\\fr_a\\rpg\\rpg_<year>_aoi.gpkg + FETCH_<year>.json (sha256, counts).
Usage: python -I fr_fetch_labels.py 2023 [2017 ...]"""
import os, sys, json, hashlib, time, urllib.parse, urllib.request
import geopandas as gpd
AOI = (0.55, 49.55, 1.10, 49.90); OUT = r'D:\sar_data\fr_a\rpg'; os.makedirs(OUT, exist_ok=True); PAGE = 5000
import re
for y in [int(a) for a in sys.argv[1:]]:
    parts, start = [], 0
    xsd = urllib.request.urlopen(f'https://data.geopf.fr/wfs/ows?SERVICE=WFS&VERSION=2.0.0&REQUEST=DescribeFeatureType&TYPENAMES=RPG.{y}:parcelles_graphiques', timeout=120).read().decode()
    GEOM = re.search(r'name="(\w+)"[^>]*type="gml:\w+PropertyType"', xsd).group(1)        # 'geom' (most years) or 'the_geom' (2022)
    while True:
        q = urllib.parse.urlencode(dict(SERVICE='WFS', VERSION='2.0.0', REQUEST='GetFeature', TYPENAMES=f'RPG.{y}:parcelles_graphiques', OUTPUTFORMAT='application/json',
                                        SRSNAME='EPSG:4326', COUNT=PAGE, STARTINDEX=start, SORTBY='id_parcel',
                                        CQL_FILTER=f"BBOX({GEOM},{AOI[0]},{AOI[1]},{AOI[2]},{AOI[3]},'EPSG:4326')"))
        for att in range(5):                                   # transient 400/5xx seen on the Geoplateforme WFS: retry
            try: raw = urllib.request.urlopen('https://data.geopf.fr/wfs/ows?' + q, timeout=600).read(); break
            except Exception as e: print(y, start, 'retry', att, repr(e)[:120], flush=True); time.sleep(10 * (att + 1))
        else: raise SystemExit(f'failed {y} {start}')
        tmp = os.path.join(OUT, f'_page_{y}_{start}.geojson'); open(tmp, 'wb').write(raw)
        g = gpd.read_file(tmp); os.remove(tmp)
        if len(g) == 0: break
        parts.append(g); start += len(g); print(y, 'fetched', start, flush=True)
        if len(g) < PAGE: break
        time.sleep(0.5)
    G = gpd.GeoDataFrame(gpd.pd.concat(parts, ignore_index=True), crs='EPSG:4326')[['id_parcel', 'code_cultu', 'surf_parc', 'geometry']]
    G = G.drop_duplicates('id_parcel').reset_index(drop=True)
    p = os.path.join(OUT, f'rpg_{y}_aoi.gpkg'); G.to_file(p, driver='GPKG')
    h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
    json.dump(dict(year=y, n=len(G), sha256=h, aoi=AOI, source='https://data.geopf.fr/wfs/ows RPG.%d:parcelles_graphiques' % y, licence='Licence Ouverte 2.0',
                   utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())), open(os.path.join(OUT, f'FETCH_{y}.json'), 'w'), indent=1)
    print(y, 'saved', len(G), h[:12], flush=True)
