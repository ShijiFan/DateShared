"""Assemble the TGRS submission package: manuscript, supplement and cover-letter PDFs, plus a compilable LaTeX source zip that contains
only the files actually used (\\input / \\includegraphics traced from main.tex, supplement.tex, cover_letter.tex). Writes SHA-256 sums.
Usage: python make_submission.py  (after compiling the three PDFs)."""
import hashlib, re, shutil, zipfile
from pathlib import Path
HERE = Path(__file__).resolve().parent; OUT = HERE / 'submission_TGRS'; OUT.mkdir(exist_ok=True)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def trace(tex, seen):
    if tex in seen: return
    seen.add(tex); s = tex.read_text(encoding='utf-8-sig')
    for m in re.findall(r'\\input\{([^}]+)\}', s):
        p = HERE / (m if m.endswith('.tex') else m + '.tex'); assert p.exists(), f'missing input {m}'; trace(p, seen)


texs = set()
for t in ('main.tex', 'supplement.tex', 'cover_letter.tex'): trace(HERE / t, texs)
figs = set()
for t in texs:
    for m in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', t.read_text(encoding='utf-8-sig')):
        base = m if m.startswith('figs/') else 'figs/' + m
        cand = [HERE / base] if Path(base).suffix else [HERE / (base + e) for e in ('.pdf', '.png', '.jpg')]
        hit = [c for c in cand if c.exists()]; assert hit, f'missing figure {m}'; figs.add(hit[0])
src = OUT / 'TGRS_source.zip'
with zipfile.ZipFile(src, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in sorted(texs | figs): z.write(f, f.relative_to(HERE).as_posix())
    for extra in ('refs.bib', 'main.bbl', 'supplement.bbl'):
        if (HERE / extra).exists(): z.write(HERE / extra, extra)
    z.writestr('README.txt', 'Compile main.tex, supplement.tex and cover_letter.tex with pdfLaTeX + BibTeX (IEEEtran.cls, IEEEtran.bst).\n'
                             'Figures are in figs/ (\\graphicspath{{figs/}}); tables in tables/.\n')
for a, b in {'main.pdf': 'Manuscript_TGRS.pdf', 'supplement.pdf': 'Supplementary_Material.pdf', 'cover_letter.pdf': 'Cover_Letter.pdf'}.items():
    shutil.copy2(HERE / a, OUT / b)
lines = [f'{sha(p)}  {p.name}' for p in sorted(OUT.iterdir()) if p.suffix in ('.pdf', '.zip')]
(OUT / 'SHA256SUMS.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('\n'.join(lines)); print('tex files:', len(texs), 'figures:', len(figs))
