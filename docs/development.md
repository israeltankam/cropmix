# Development

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev,viz,docs]"
```

## Tests

```bash
pytest
ruff check .
```

## Documentation

```bash
mkdocs serve
```

## Build distributions

```bash
python -m build
python -m twine check dist/*
```

## Scientific contribution rule

A change to epidemic machinery should include:

- the mathematical event/rate;
- its units;
- the altered biological assumption;
- at least one invariant, limiting-case or regression test.
