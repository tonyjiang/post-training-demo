"""Download pinned original weights; verify sizes and upstream hashes before publication."""
import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='configs/base-model.json')
    parser.add_argument('--output', default='/root/models/Olmo-3-1025-7B')
    parser.add_argument('--endpoint', default='https://huggingface.co')
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)

    def verify(path, entry):
        if not path.exists() or path.stat().st_size != entry['size']:
            return False
        algorithm = 'sha256' if 'lfs' in entry else 'sha1'
        digest = hashlib.new(algorithm)
        if algorithm == 'sha1':
            digest.update(f"blob {entry['size']}\0".encode())
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        expected = entry['lfs']['sha256'] if 'lfs' in entry else entry['blobId']
        return digest.hexdigest() == expected

    def download(entry):
        name = entry['rfilename']
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if verify(path, entry):
            print(f'Already verified: {name}', flush=True)
            return
        partial = path.with_name(path.name + '.partial')
        prefix = 'datasets/' if manifest.get('repo_type') == 'dataset' else ''
        url = f"{args.endpoint}/{prefix}{manifest['repo_id']}/resolve/{manifest['revision']}/{name}"
        print(f'Downloading: {name} ({entry["size"]} bytes)', flush=True)
        subprocess.run(['curl', '-4', '-fLsS', '--retry', '8', '--retry-all-errors',
                        '--connect-timeout', '20', '--speed-limit', '1024', '--speed-time', '120',
                        '-C', '-', '-o', str(partial), url], check=True)
        if not verify(partial, entry):
            raise RuntimeError(f'Size/hash mismatch: {partial}; remove partial before retrying')
        partial.replace(path)
        print(f'Verified: {name}', flush=True)

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(download, manifest['files']))
    (destination / 'download-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('All files verified.', flush=True)


if __name__ == '__main__':
    main()
