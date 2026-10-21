PY ?= python
export PYTHONPATH := src

.PHONY: all ingest test clean
all:
	$(PY) -m curvehedge.pipeline
ingest:
	$(PY) -m curvehedge.ingest
test:
	$(PY) -m pytest -q tests
clean:
	rm -rf reports/figures/*.png reports/*.csv reports/metrics.json
