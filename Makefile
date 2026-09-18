PYTHON ?= python

.PHONY: all pipeline pipeline-skip-ingest install test clean help

all: pipeline

help:
	@echo "Available targets:"
	@echo "  install               Install Python dependencies"
	@echo "  test                  Run unit tests using pytest"
	@echo "  pipeline              Run the full pipeline (ingest + features + labels + scaling)"
	@echo "  pipeline-skip-ingest  Run the pipeline skipping CSV ingestion"
	@echo "  clean                 Clean pycache and temporary build artifacts"

install:
	pip install -r requirements.txt

test:
	pytest

pipeline:
	$(PYTHON) scripts/run_pipeline.py

pipeline-skip-ingest:
	$(PYTHON) scripts/run_pipeline.py --skip-ingest

clean:
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__') if p.is_dir()]"