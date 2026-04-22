# Runbook — doc-intelligence-rag-assistant

## Start the system
```powershell
# 1. Start infrastructure
docker-compose up -d pgvector redis

# 2. Activate venv
.\venv\Scripts\Activate.ps1

# 3. Start API
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

# 4. Start UI (separate terminal)
streamlit run src/ui/streamlit_app.py
```

## Run ingestion
```powershell
python scripts/ingest.py --data-dir data/raw/
```

## Run evaluation
```powershell
python scripts/evaluate.py
```

## Run tests
```powershell
pytest tests/ -v
```

## Check costs
```powershell
python -c "from src.monitoring.cost_monitor import CostMonitor; CostMonitor().report()"
```

## Emergency: suspend LLM calls
In .env, set: `ANTHROPIC_API_KEY=SUSPENDED`