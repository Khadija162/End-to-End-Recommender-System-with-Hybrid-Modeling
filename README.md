# End-to-End Recommender System with Hybrid Modeling

A local **full-stack hybrid recommendation prototype** with content-based TF-IDF similarity, user-user collaborative ratings, Bayesian popularity fallback for cold-start users, FastAPI, Streamlit, an open CC0 dataset, validation and Bronze/Silver/Gold ETL, Docker, pytest and GitHub Actions.

> **Maturity:** Source code committed; Docker startup, actual CI results and production recommendation quality have not yet been independently verified. Ranking scores are **not probabilities**.

## Architecture

```text
Open CC0 items + ratings
     |
CSV validation and quality checks
     +--> Bronze CSV -> Silver validated/rejected CSV -> Gold popularity metrics
     |
Content model: TF-IDF title + category + description
Collaborative model: user-user rating similarity
Cold-start baseline: Bayesian-smoothed item popularity
     |
Weighted hybrid ranking -> exclude seen items
     |
FastAPI /api/recommendations/{user_id}
     |
Streamlit recommendation explorer
```

## Open dataset

`data/public/items.csv` has **12 original fictional learning resources**, and `data/public/interactions.csv` has **24 ratings from six synthetic users**, released under **CC0 1.0**. No real people or private preference records are included. Read [provenance](data/public/README.md). These small samples are appropriate for demonstration but insufficient for real recommendation performance claims.

## Windows PowerShell quick start

```powershell
git clone https://github.com/Khadija162/End-to-End-Recommender-System-with-Hybrid-Modeling.git
cd End-to-End-Recommender-System-with-Hybrid-Modeling
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m recommender.pipeline
.\.venv\Scripts\python.exe -m pytest -q
```

Launch backend and frontend in separate terminals:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py
```

Or start Docker Desktop and run:

```powershell
docker compose up --build
```

API docs: http://localhost:8000/docs
Dashboard: http://localhost:8501

## API

- `GET /health`: service health
- `GET /api/items`: item catalog
- `GET /api/users`: available demo users
- `GET /api/recommendations/u1?limit=5&content_weight=0.55`: personalized ranking
- `GET /api/recommendations/new-user`: cold-start baseline

The content weight adjusts the content/collaborative components of ranking. Existing items rated by that user are excluded. `rank_score` is an uncalibrated signal for sorting, not a probability.

## Current limitations and next work

The API currently loads source CSV fixtures, not a deployed trained model or an approved versioned Silver feature dataset. This is a deterministic **hybrid heuristic** demonstration rather than a large-scale trained recommender or online learning system. Add real, appropriately licensed ratings such as MovieLens with GroupLens attribution, temporal holdout metrics (Recall@K / NDCG@K), popularity debiasing, implicit signals, a feature store, event ingestion and retraining. Add authentication and privacy controls before serving real user data.

**Deployment:** Docker Compose is configured locally; no cloud deployment is included or claimed. Tests and Docker runs still need verification on your machine or in GitHub Actions.
