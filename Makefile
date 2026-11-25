.PHONY: help install lint format test pipeline mlflow serve up down simulate simulate-drift drift research research-resume research-quick clean

help:            ## Show available commands
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

install:         ## Install dev dependencies (run inside your venv)
	pip install -r requirements/dev.txt && pip install -e . --no-deps

lint:            ## Lint + format check
	ruff check src tests scripts && ruff format --check src tests scripts

format:          ## Auto-fix lint + format
	ruff check --fix src tests scripts && ruff format src tests scripts

test:            ## Run tests with coverage
	pytest --cov=churn --cov-report=term-missing

pipeline:        ## Run the production DVC pipeline (only changed stages re-run)
	dvc repro dvc.yaml && dvc metrics show reports/metrics.json

mlflow:          ## Open the MLflow UI at http://localhost:5050
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5050

serve:           ## Run the API locally with hot reload at http://localhost:8080/docs
	PREDICTION_LOG_PATH=logs/predictions.jsonl uvicorn churn.api.main:app --reload --port 8080

up:              ## Start API + Prometheus + Grafana in Docker
	docker compose up -d --build

down:            ## Stop the Docker stack
	docker compose down

simulate:        ## Send 300 normal requests to the API
	python scripts/simulate_traffic.py --n 300

simulate-drift:  ## Send 300 drifted requests to the API
	python scripts/simulate_traffic.py --n 300 --drift

drift:           ## Check live traffic for data drift
	python -m churn.monitoring.drift

research:        ## Run the full drift study (~70 min): experiments -> tables -> figures
	dvc repro research/dvc.yaml

research-resume: ## Resume an interrupted study (skips finished stages, keeps the Mac awake)
	caffeinate -i python -m driftlab.run all --skip-existing 2>&1 \
		| grep --line-buffered -v "HTTP Request\|resource_tracker\|warnings.warn" \
		| tee -a research/results/run.log
	python -m driftlab.analysis && python -m driftlab.plots

research-quick:  ## 5-minute smoke run of every experiment (not for reporting)
	python -m driftlab.run all --quick && python -m driftlab.analysis && python -m driftlab.plots

clean:           ## Remove caches and logs
	rm -rf .pytest_cache .ruff_cache .coverage logs/ reports/drift_report.json
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
