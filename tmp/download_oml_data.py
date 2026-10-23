"""Download source data only; never execute repository code."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

ROOT = Path(r'E:\lithography_machine_lens\film_hppo\outputs\references\oml_ppo')
REPO = 'hammer-wang/oml-ppo'

def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'film-optical-data-conversion'})
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read()

def main():
    tree = json.loads(fetch(f'https://api.github.com/repos/{REPO}/git/trees/master?recursive=1'))
    commit = tree['sha']
    extra = {'README.md', 'LICENSE', 'rl-multilayer/RLMultilayer/utils.py',
             'rl-multilayer/RLMultilayer/taskenvs/tasks.py', 'expconfigs.py'}
    files = [x['path'] for x in tree['tree'] if x['type'] == 'blob'
             and (x['path'].startswith('data/') or x['path'] in extra)]
    def download(path):
        url = f'https://raw.githubusercontent.com/{REPO}/{commit}/' + urllib.parse.quote(path)
        raw = fetch(url)
        dest = ROOT / 'source' / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.read_bytes() != raw:
            raise FileExistsError(f'Different existing source: {dest}')
        dest.write_bytes(raw)
        return {'path': path, 'url': url, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(download, files))
    manifest = {'repository': f'https://github.com/{REPO}', 'commit': commit, 'files': records}
    (ROOT / 'source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'root': str(ROOT), 'commit': commit, 'downloaded': len(records)}))

if __name__ == '__main__':
    main()
