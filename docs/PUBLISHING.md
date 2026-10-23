# Installing remotely and publishing

The project is already a normal Python package:

- distribution name: **`emg-speech`**
- import name: **`emg_speech`**
- version: **`1.0.0`**
- Python: **3.10+**

## Project layout

The package follows the `src/` layout recommended by the
[Python Packaging User Guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/):

```text
.
├── LICENCE
├── MANIFEST.in
├── README.md
├── pyproject.toml
├── record.py
├── train.py
├── recognize.py
├── simulate.py
├── _bootstrap.py
├── src/
│   └── emg_speech/
│       ├── __init__.py
│       ├── api.py
│       └── ...
└── tests/
```

`setuptools` is configured with `package-dir = {"" = "src"}` and
`where = ["src"]`, so only the library under `src/` is packaged.  The existing
`LICENCE` spelling is matched by `license-files = ["LICEN[CS]E*"]`.

## Install from a Git repository

Push the project to GitHub/GitLab, then on any machine:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install "git+https://github.com/<your-user>/<your-repo>.git"
```

After installation, the commands are available from anywhere:

```bash
emg-record --help
emg-train --help
emg-recognize --help
emg-simulate --help
emg-demo-data --out dataset_demo
```

The old names `emg-acquire` and `emg-infer` remain as aliases.

## Install from PyPI

After the first upload:

```bash
pip install emg-speech
```

## Build and test the package locally

```bash
python -m pip install -e ".[dev]"
python -m build
twine check dist/*
```

Test the wheel in a clean environment before uploading:

```bash
python -m venv /tmp/emg-test
source /tmp/emg-test/bin/activate
pip install dist/emg_speech-1.0.0-py3-none-any.whl
emg-record --help
emg-simulate --help
```

## Publish to PyPI

1. Create accounts on <https://pypi.org> and <https://test.pypi.org>.
2. Create an API token for the project.
3. Build and check:

```bash
rm -rf dist build
python -m build
twine check dist/*
```

4. Upload to TestPyPI first:

```bash
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ emg-speech
```

5. Upload to the real PyPI:

```bash
twine upload dist/*
```

Use `__token__` as the username and paste the API token as the password.

## Before publishing

The author and MIT license are already filled in, and `LICENCE` contains the
MIT text.  Before uploading:

- replace `YOUR_GITHUB_USER` in `[project.urls]` with your repository account
- choose the next `version` for every upload (PyPI rejects duplicate versions)
- check that the `authors` entry still matches the copyright holder in `LICENCE`

`license = "MIT"` and `license-files = ["LICEN[CS]E*"]` use the modern PEP 639
metadata supported by `setuptools>=77.0.3`.

The heavy dependencies (`torch` and `PyQt6`) are installed by default because
the desktop apps need them.  If you later want a lighter core-only install,
split them into optional extras, for example:

```toml
[project.optional-dependencies]
gui = ["PyQt6>=6.5"]
train = ["torch>=2.1"]
all = ["PyQt6>=6.5", "torch>=2.1"]
```

The package includes only Python files, so no extra package-data configuration
is required.
