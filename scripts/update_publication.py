"""Synchronize publication links after an arXiv ID is assigned."""
from pathlib import Path
import argparse
import json
import re

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--arxiv-id', help='Assigned arXiv ID, e.g. YYMM.NNNNN')
args = parser.parse_args()
path = root / 'publication.json'
cfg = json.loads(path.read_text(encoding='utf-8'))
if args.arxiv_id:
    if not re.fullmatch(r'\d{4}\.\d{4,5}', args.arxiv_id):
        parser.error('Expected a real arXiv ID in YYMM.NNNNN format')
    cfg['arxiv_id'] = args.arxiv_id
    path.write_text(json.dumps(cfg, indent=2)+'\n', encoding='utf-8')
aid = cfg.get('arxiv_id')
arxiv = 'https://arxiv.org/abs/' + aid if aid else None
hf = 'https://huggingface.co/papers/' + aid if aid else None
links = [f'[Paper]({cfg["project"]}static/pdf/VERSE.pdf)',
         f'[Project page]({cfg["project"]})', f'[Code]({cfg["github"]})']
if aid:
    links += [f'[arXiv]({arxiv})', f'[Hugging Face Paper]({hf})']
if cfg.get('model'):
    links += [f'[VERSE-7B Model]({cfg["model"]})']
block = ' · '.join(links)
if not aid:
    block += '\n\narXiv and Hugging Face Paper links will be added after the arXiv ID is assigned.'
readme = root / 'README.md'
text = readme.read_text(encoding='utf-8')
text = re.sub(r'<!-- publication:start -->.*?<!-- publication:end -->',
    '<!-- publication:start -->\n' + block + '\n<!-- publication:end -->', text, flags=re.S)
readme.write_text(text, encoding='utf-8')
authors = [('Xinran','Liu'),('Yilei','Jiang'),('Tianshuo','Peng'),('Kaixuan','Fan'),('Jinyuan','Hu'),('Xiangyu','Yue')]
bib = ('@misc{liu2026verse,\n  title={' + cfg['title'] + '},\n  author={' +
       ' and '.join(last + ', ' + first for first,last in authors) + '},\n  year={2026},\n')
if aid:
    bib += '  eprint={' + aid + '},\n  archivePrefix={arXiv},\n'
bib += '  url={' + (arxiv or cfg['project']) + '}\n}'
(root / 'CITATION.bib').write_text(bib+'\n', encoding='utf-8')
cff = ['cff-version: 1.2.0', 'message: "Please cite the VERSE paper when using this work."',
       'type: software', 'title: ' + json.dumps(cfg['title']), 'authors:']
for first,last in authors:
    cff += ['  - family-names: ' + last, '    given-names: ' + first]
cff += ['repository-code: ' + cfg['github'], 'url: ' + cfg['project'], 'license: MIT']
if aid:
    cff += ['identifiers:', '  - type: url', '    value: ' + arxiv]
(root / 'CITATION.cff').write_text('\n'.join(cff)+'\n', encoding='utf-8')
data = dict(cfg, arxiv=arxiv, hf_paper=hf, bibtex=bib)
js = 'const publication = ' + json.dumps(data, ensure_ascii=False) + ';\n' + '''
(function () {
  const root = document.getElementById('publication-links');
  const links = [['Paper', 'static/pdf/VERSE.pdf', 'fas fa-file-pdf'],
    ['Code', publication.github, 'fab fa-github']];
  if (publication.arxiv) links.push(['arXiv', publication.arxiv, 'ai ai-arxiv']);
  if (publication.hf_paper) links.push(['Hugging Face Paper', publication.hf_paper, 'fas fa-file-lines']);
  if (publication.model) links.push(['VERSE-7B Model', publication.model, 'fas fa-cube']);
  links.push(['Rollouts', '#demo', 'fas fa-images']);
  links.forEach(function (item) {
    const span = document.createElement('span'); span.className = 'link-block';
    const a = document.createElement('a'); a.href = item[1];
    a.className = 'button is-normal is-rounded is-dark';
    if (!item[1].startsWith('#')) { a.target = '_blank'; a.rel = 'noopener'; }
    const icon = document.createElement('span'); icon.className = 'icon';
    const i = document.createElement('i'); i.className = item[2]; icon.appendChild(i);
    const label = document.createElement('span'); label.textContent = item[0];
    a.append(icon, label); span.appendChild(a); root.appendChild(span);
  });
  document.getElementById('bibtex').textContent = publication.bibtex;
})();
'''
(root / 'docs' / 'static' / 'js' / 'publication.js').write_text(js, encoding='utf-8')
print('Synchronized README, project page, and citations.' + (' arXiv ID: '+aid if aid else ' arXiv ID not yet assigned.'))
