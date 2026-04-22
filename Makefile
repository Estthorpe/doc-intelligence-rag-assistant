.PHONY: setup test type-check lint freeze check run ui ingest evaluate help

setup:
	pip install -r requirements.txt
	python -m spacy download en_core_web_lg
	pre-commit install

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

type-check:
	mypy src/ --strict

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-scenarios:
	pytest tests/scenarios/ -v

freeze:
	pip freeze | grep -v "pywin32" | grep -v "pywin32-ctypes" | grep -v "pywinpty" > requirements.txt
	@echo "requirements.txt updated (Windows packages stripped)"

check: lint type-check test
	@echo "All checks passed. Safe to push."

run:
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

ui:
	streamlit run src/ui/streamlit_app.py

ingest:
	python scripts/ingest.py --data-dir data/raw/

evaluate:
	python scripts/evaluate.py

help:
	@echo "make setup      - Install all dependencies"
	@echo "make lint       - Run ruff linter and format check"
	@echo "make type-check - Run mypy strict"
	@echo "make test       - Run all tests with coverage"
	@echo "make freeze     - Update requirements.txt (strips Windows packages)"
	@echo "make check      - Run all quality gates (lint + type-check + test)"
	@echo "make run        - Start FastAPI server on port 8000"
	@echo "make ui         - Start Streamlit UI on port 8501"
	@echo "make ingest     - Run document ingestion pipeline"
	@echo "make evaluate   - Run RAGAS evaluation against golden set"