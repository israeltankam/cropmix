# Hosting and releasing Cropmix

This guide takes Cropmix from the repository in this folder to a package that users can install with:

```bash
python -m pip install cropmix
```

The recommended stack is GitHub + GitHub Actions + PyPI Trusted Publishing + GitHub Pages + Zenodo.

---

## 1. Replace repository placeholders

Search the repository for:

```text
YOUR_GITHUB_USERNAME
```

and replace it in:

- `pyproject.toml`
- `mkdocs.yml`
- `CITATION.cff`

Also review the package author, license, project description, and URLs before the first public release.

### Package name

The Python distribution name in `pyproject.toml` is:

```toml
name = "cropmix"
```

PyPI project names are globally unique. Check `https://pypi.org/project/cropmix/` immediately before publication. A missing page today does not reserve the name for you.

---

## 2. Create the Git repository locally

From the repository root:

```bash
git init
git add .
git commit -m "Initial Cropmix alpha"
git branch -M main
```

Create a new empty GitHub repository named `cropmix`, then connect it:

```bash
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/cropmix.git
git push -u origin main
```

Do not upload an existing `.venv`, `dist/`, or secrets. They are already covered by `.gitignore`.

---

## 3. Validate locally before publishing anything

Create a clean virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[dev,viz,docs]"
```

Run:

```bash
pytest
ruff check .
mkdocs build --strict
```

Then build the package:

```bash
rm -rf dist build
python -m build
python -m twine check dist/*
```

You should obtain both a source distribution and a wheel in `dist/`.

Test the wheel in a second clean environment rather than only testing the editable checkout:

```bash
python -m venv /tmp/cropmix-wheel-test
source /tmp/cropmix-wheel-test/bin/activate
python -m pip install dist/cropmix-0.1.0-py3-none-any.whl
python -c "import cropmix; print(cropmix.__version__)"
cropmix doctor
```

---

## 4. Let GitHub Actions run CI

`.github/workflows/ci.yml` runs the test suite on supported Python versions for pushes and pull requests.

After the first push, open:

```text
GitHub repository → Actions → CI
```

Do not publish to PyPI until CI is green.

---

## 5. Optional but recommended: test on TestPyPI

Create an account at TestPyPI if you do not already have one.

You can upload a local build with Twine:

```bash
python -m twine upload --repository testpypi dist/*
```

Then test installation in a fresh environment:

```bash
python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  cropmix
```

The extra PyPI index is useful because Cropmix's ordinary dependencies are not necessarily mirrored on TestPyPI.

If you prefer, configure TestPyPI Trusted Publishing in the same manner as production PyPI instead of storing an API token.

---

## 6. Configure production PyPI Trusted Publishing

PyPI Trusted Publishing uses GitHub's OpenID Connect identity. It avoids storing a long-lived PyPI token in repository secrets.

The included release workflow is:

```text
.github/workflows/release.yml
```

and its protected GitHub environment is named:

```text
pypi
```

### On PyPI

Sign in to PyPI and configure a GitHub Trusted Publisher for the project. For a first publication you can use PyPI's pending-publisher workflow.

Use these values:

```text
PyPI project name: cropmix
GitHub owner: YOUR_GITHUB_USERNAME
GitHub repository: cropmix
Workflow filename: release.yml
Environment: pypi
```

The workflow path is the filename, not `.github/workflows/release.yml`, in the PyPI form.

### On GitHub

Go to:

```text
Settings → Environments → New environment → pypi
```

For a scientific package, configure a required reviewer if practical. This prevents an accidental release event from publishing immediately.

No `PYPI_API_TOKEN` secret is required by the supplied workflow.

---

## 7. Make the first release

Update the version in:

- `pyproject.toml`
- `src/cropmix/__init__.py`
- `CITATION.cff`
- `CHANGELOG.md`

Commit and push:

```bash
git add .
git commit -m "Release 0.1.0"
git push
```

Then create a GitHub release:

```text
GitHub → Releases → Draft a new release
```

Create tag:

```text
v0.1.0
```

Publish the release.

The `release.yml` workflow will:

1. check out the tagged source;
2. build wheel + sdist;
3. validate distributions;
4. upload build artifacts between jobs;
5. request a short-lived OIDC publishing credential;
6. publish to PyPI through `pypa/gh-action-pypi-publish`.

After the workflow completes, test:

```bash
python -m venv /tmp/cropmix-pypi-test
source /tmp/cropmix-pypi-test/bin/activate
python -m pip install cropmix
python -c "import cropmix; print(cropmix.__version__)"
```

---

## 8. Publish the documentation with GitHub Pages

The supplied workflow is:

```text
.github/workflows/docs.yml
```

It builds the MkDocs site and deploys it with GitHub Pages actions.

In GitHub:

```text
Settings → Pages → Build and deployment → Source → GitHub Actions
```

Then push to `main` or manually run the docs workflow.

The default URL will be approximately:

```text
https://YOUR_GITHUB_USERNAME.github.io/cropmix/
```

Once live, make sure the `Documentation` URL in `pyproject.toml` points there.

To preview locally:

```bash
mkdocs serve
```

---

## 9. Connect the repository to Zenodo

Zenodo can archive each GitHub release and mint a DOI.

1. Sign in to Zenodo.
2. Connect your GitHub account.
3. Find the `cropmix` repository in the Zenodo GitHub integration.
4. Enable the repository.
5. Create a GitHub release.
6. Wait for Zenodo to ingest it.
7. Copy the DOI into the README/docs when appropriate.

`CITATION.cff` is already included and should be kept synchronized with releases.

Use the Zenodo concept DOI for citing the software family and version DOIs when exact reproducibility of a specific release matters.

---

## 10. EpiPvr on users' machines

`pip install cropmix` does **not** install R. This is intentional.

Core simulation, mean-field analysis, calibration, and optimization work entirely in Python.

Users who want EpiPvr inference need to install:

1. R;
2. a working compilation toolchain appropriate for their platform if needed by R/Stan packages;
3. EpiPvr in R:

```r
install.packages("EpiPvr")
```

Then:

```bash
cropmix doctor
```

Cropmix calls `Rscript` internally. Users do not need to open R during ordinary Cropmix analyses.

Do not vendor an entire R installation into the PyPI wheel.

---

## 11. Release discipline

A useful version scheme is semantic versioning:

```text
0.1.x  API is alpha; bug fixes and small scientific corrections
0.2.0  substantial new capability, e.g. posterior propagation improvements
0.x    API may still change
1.0.0  first stable scientific/public API
```

Before each release:

```bash
pytest
ruff check .
mkdocs build --strict
python -m build
python -m twine check dist/*
```

Also verify:

- package version;
- changelog;
- citation metadata;
- documentation examples;
- numerical regression tests;
- the EpiPvr bridge against the CRAN version you support;
- whether model assumptions changed.

---

## 12. What should be hosted where?

| Asset | Recommended home |
|---|---|
| Source code | GitHub |
| Installable Python distributions | PyPI |
| User/manual/API documentation | GitHub Pages |
| Tagged release source archive + DOI | Zenodo |
| Large benchmark datasets | Zenodo or another research-data repository |
| CI logs/tests | GitHub Actions |
| Issues and feature requests | GitHub Issues |

Avoid committing large Monte Carlo outputs directly to the Git repository.

---

## 13. Suggested first public release sequence

Use this order:

```text
1. Replace placeholders
2. Run all local tests
3. Push GitHub repository
4. Confirm GitHub CI
5. Build docs and enable GitHub Pages
6. Test wheel locally
7. TestPyPI dry run
8. Configure PyPI Trusted Publisher
9. Enable Zenodo repository integration
10. Create GitHub v0.1.0 release
11. Verify PyPI installation
12. Verify Zenodo archival and docs links
```

This gives users one consistent path:

```bash
python -m pip install cropmix
```

and one canonical documentation site.
