# Building the docs

## Dependencies

Install the doc dependencies into your environment:

```bash
pip install sphinx sphinx-book-theme myst-nb numpydoc
```

## Build

From the repo root:

```bash
python -m sphinx -b html docs/source docs/build/html
```

The output is written to `docs/build/html/`. Open `docs/build/html/index.html` in a browser to view it.

To do a clean rebuild, delete `docs/build/` first:

```bash
rm -rf docs/build/
python -m sphinx -b html docs/source docs/build/html
```
