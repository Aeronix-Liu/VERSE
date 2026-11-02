const publication = {"title": "VERSE: Verifiable Spatial Self-Evolution with Geometric Process Rewards", "arxiv_id": null, "github": "https://github.com/Aeronix-Liu/VERSE", "project": "https://aeronix-liu.github.io/VERSE/", "year": 2026, "arxiv": null, "hf_paper": null, "bibtex": "@misc{liu2026verse,\n  title={VERSE: Verifiable Spatial Self-Evolution with Geometric Process Rewards},\n  author={Liu, Xinran and Jiang, Yilei and Peng, Tianshuo and Fan, Kaixuan and Hu, Jinyuan and Yue, Xiangyu},\n  year={2026},\n  url={https://aeronix-liu.github.io/VERSE/}\n}"};

(function () {
  const root = document.getElementById('publication-links');
  const links = [['Paper', 'static/pdf/VERSE.pdf', 'fas fa-file-pdf'],
    ['Code', publication.github, 'fab fa-github']];
  if (publication.arxiv) links.push(['arXiv', publication.arxiv, 'ai ai-arxiv']);
  if (publication.hf_paper) links.push(['Hugging Face Paper', publication.hf_paper, 'fas fa-file-lines']);
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
