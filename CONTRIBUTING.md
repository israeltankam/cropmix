# Contributing to Cropmix

Cropmix is a scientific package. Changes to epidemic processes must be accompanied by:

1. a mathematical statement of the process and its rate;
2. a unit test for the relevant invariant or limiting case;
3. documentation of any changed biological assumption;
4. a changelog entry when user-visible behaviour changes.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\\Scripts\\activate
python -m pip install -U pip
python -m pip install -e ".[dev,viz,docs]"
pytest
ruff check .
```

Pull requests should keep public APIs backwards compatible within a minor release whenever practical.
