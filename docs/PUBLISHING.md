# Publication links

Public code: https://github.com/Aeronix-Liu/VERSE

Project page: https://aeronix-liu.github.io/VERSE/

The original private repository is retained at
https://github.com/Aeronix-Liu/VERSE-private . This public repository starts with
a fresh Git history. It does not include private Git history or model weights.

## After arXiv assigns an ID

Run `python scripts/update_publication.py --arxiv-id YYMM.NNNNN` using the actual
assigned ID. Commit and push the updated publication metadata. This updates the
README, project-page navigation, and BibTeX/CFF citations together.

Visit `https://huggingface.co/papers/YYMM.NNNNN` to index the paper, claim authorship,
and set its GitHub repository and project-page links to the addresses above.
No model upload is necessary to create a Hugging Face Paper page.

For arXiv's Comments field, use:

`Project page: https://aeronix-liu.github.io/VERSE/; Code: https://github.com/Aeronix-Liu/VERSE`

Upload the prepared LaTeX source ZIP separately through arXiv. Inspect the PDF
rendered by arXiv before completing submission. No arXiv ID is invented here.

## Checkpoints

The released VERSE-7B checkpoint is linked from the README and project page:
https://huggingface.co/Aeronix-zzz/VERSE-7B
No existing model repository visibility is changed by these publishing scripts.

After the arXiv ID is assigned, adding the actual arXiv URL to each model
card allows Hugging Face to associate it with the paper. Preserve each model's
existing card metadata and license.

## Project-page updates

The GitHub Actions workflow publishes `docs/` to GitHub Pages on pushes to main.
The downloadable paper lives at `docs/static/pdf/VERSE.pdf`.
