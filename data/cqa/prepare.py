"""Restore the original CQA document files using only Python's standard library."""
import hashlib
import json
from pathlib import Path
import tempfile


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    directory = Path(__file__).resolve().parent
    manifest = json.loads((directory / 'documents.manifest.json').read_text(encoding='utf-8'))
    for name, info in manifest.items():
        target = directory / name
        if target.exists() and fingerprint(target) == info['sha256']:
            print(f'OK: {name} (already verified)')
            continue
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=directory, suffix='.tmp', delete=False) as output:
                temporary = Path(output.name)
                digest = hashlib.sha256()
                size = count = 0
                for part in info['parts']:
                    with (directory / part).open('rb') as source:
                        for line in source:
                            # Git may check out text with LF or CRLF on different OSes.
                            line = line.replace(b'\r\n', b'\n')
                            if info['newline'] == 'CRLF':
                                line = line.replace(b'\n', b'\r\n')
                            output.write(line)
                            digest.update(line)
                            size += len(line)
                            count += 1
                if (digest.hexdigest() != info['sha256']
                        or size != info['bytes'] or count != info['lines']):
                    raise ValueError(f'Integrity check failed for {name}; restore the part files from Git.')
            temporary.replace(target)
            print(f'OK: {name} ({count} documents, SHA-256 verified)')
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
