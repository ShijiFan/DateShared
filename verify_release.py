"""Verify every released file without loading data or third-party libraries."""
import csv
import hashlib
from pathlib import Path, PurePosixPath
import sys

def main():
    root = Path(__file__).resolve().parent
    manifest = root / 'MANIFEST_SHA256.tsv'
    failures = []
    seen = set()
    with manifest.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            rel = row['path']
            parts = PurePosixPath(rel)
            if parts.is_absolute() or '..' in parts.parts or rel in seen:
                failures.append(f'invalid or duplicate path: {rel}')
                continue
            seen.add(rel)
            path = root.joinpath(*parts.parts)
            if not path.is_file():
                failures.append(f'missing: {rel}')
                continue
            data = path.read_bytes()
            if len(data) != int(row['bytes']) or hashlib.sha256(data).hexdigest().lower() != row['sha256'].lower():
                failures.append(f'content mismatch: {rel}')
    if failures:
        print('\n'.join(failures))
        print(f'FAIL: {len(failures)} issue(s) across {len(seen)} manifest entries.')
        return 1
    if not seen:
        print('FAIL: empty manifest.')
        return 1
    print(f'PASS: {len(seen)} files match MANIFEST_SHA256.tsv (SHA256 and byte count).')
    return 0

if __name__ == '__main__':
    sys.exit(main())
