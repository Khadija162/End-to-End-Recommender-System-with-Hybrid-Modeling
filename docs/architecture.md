# Architecture and limitations

Implemented: original CC0 item and interaction datasets, deterministic Bronze/Silver/Gold CSV ETL with data validation, TF-IDF item content similarity, user-user rating similarity for shared observations, Bayesian popularity smoothing for cold start, FastAPI, Streamlit, Docker and pytest CI.

Scores are heuristic rankings, not probabilities or calibrated predicted ratings. User-neighborhood similarities on very few overlapping items are fragile. The current API loads the source dataset, not the Silver tables; run the ETL separately for governance and quality evidence. Real enterprise ranking should consume a versioned validated model artifact and trained feature store.

Not implemented: production identity, real interactions, online event tracking, ANN retrieval, vector databases, rigorous offline held-out recommendation metrics, experimentation, freshness monitoring, scale tests or deployment. Explore the GroupLens MovieLens datasets with appropriate licensing and attribution for richer evaluation.
