# CO2 Forecast — dev / build / serve helpers
#
# Assumes:
#   - conda env `py39_pt` with backend/requirements.txt installed
#   - Node 18 + npm 10 on PATH (via nvm)
#
# Usage:
#   make dev-backend       # FastAPI @ :8000 with reload
#   make dev-frontend      # Vite   @ :5173 with /api proxy → :8000
#   make build-frontend    # frontend/dist → backend/static
#   make serve             # build + run FastAPI serving built frontend (single port :8000)

CONDA_ACTIVATE := source $(HOME)/miniconda3/etc/profile.d/conda.sh && conda activate py39_pt

.PHONY: help dev-backend dev-frontend install-backend install-frontend build-frontend serve clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

install-backend: ## pip install backend requirements into py39_pt
	$(CONDA_ACTIVATE) && pip install -r backend/requirements.txt

install-frontend: ## npm install in frontend/
	cd frontend && npm install

dev-backend: ## Run FastAPI with auto-reload on :8000
	$(CONDA_ACTIVATE) && cd backend && uvicorn app:app --reload --host 127.0.0.1 --port 8000

dev-frontend: ## Run Vite dev server on :5173 (proxies /api → :8000)
	cd frontend && npm run dev

build-frontend: ## Build frontend into backend/static
	cd frontend && npm run build

serve: build-frontend ## Build frontend and serve everything on :8000
	$(CONDA_ACTIVATE) && cd backend && uvicorn app:app --host 0.0.0.0 --port 8000

test-backend: ## Run backend pytest smoke suite
	$(CONDA_ACTIVATE) && cd backend && pytest

types: ## Regenerate frontend/src/api/schema.d.ts from FastAPI /openapi.json
	$(CONDA_ACTIVATE) && cd backend && python -c "import json; from app import app; json.dump(app.openapi(), open('../frontend/openapi.json','w'), indent=2)"
	cd frontend && npx openapi-typescript openapi.json -o src/api/schema.d.ts

clean: ## Remove build artifacts and caches
	rm -rf frontend/dist backend/static backend/__pycache__
	find backend -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
