PYTHON ?= python3

.PHONY: all pipeline pipeline-skip-ingest install test

all: pipeline

install:
	pip install -r requirements.txt

test:
	pytest

pipeline:
	$(PYTHON) scripts/run_pipeline.py

pipeline-skip-ingest:
	$(PYTHON) scripts/run_pipeline.py --skip-ingest
