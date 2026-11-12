PY ?= python

.PHONY: all fit analyze test smoke install pymc-install pymc-fit pymc-compare

## Fit the model, then regenerate every table, chart and number in reports/.
all: fit analyze

## Full MCMC fit (4 chains x 1,000 kept draws); writes artifacts/model.binpb and reports/fit_info.json.
fit:
	$(PY) scripts/fit_model.py

analyze:
	$(PY) scripts/analyze.py

test:
	$(PY) -m pytest -q

## End-to-end run with a few MCMC draws; checks the code path, not the numbers.
smoke:
	$(PY) scripts/fit_model.py --quick --out smoke/model.binpb
	$(PY) scripts/analyze.py --model smoke/model.binpb --out-dir smoke/reports --skip-matrix

install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e . --no-deps

## PyMC-Marketing cross-check (separate environment: Python 3.12+, requirements-pymc.txt)
PYMC ?= .venv-pymc/bin/python

pymc-install:
	python3 -m venv .venv-pymc && $(PYMC) -m pip install -r requirements-pymc.txt

## Needs reports/pymc/meridian_knots.json (scripts/export_meridian_knots.py, Meridian environment).
pymc-fit:
	$(PYMC) scripts/fit_pymc.py

pymc-compare:
	$(PYMC) scripts/compare_pymc.py
