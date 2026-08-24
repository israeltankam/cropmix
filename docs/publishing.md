# Publishing and hosting

The repository root contains a detailed `HOSTING.md` guide.

The intended production stack is:

- **GitHub** — source, issues, pull requests, CI;
- **GitHub Actions** — tests, package build, release publication, docs deployment;
- **PyPI** — `pip install cropmix`;
- **GitHub Pages** — documentation site;
- **Zenodo** — archived releases and DOI.

Before the first release replace every `YOUR_GITHUB_USERNAME` placeholder in `pyproject.toml`, `mkdocs.yml`, and `CITATION.cff`.
