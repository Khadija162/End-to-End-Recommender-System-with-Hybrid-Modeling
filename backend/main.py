"""Recommendation API; demo only, no production identity controls."""
import os
from functools import lru_cache
from fastapi import FastAPI, HTTPException, Query
import pandas as pd
from recommender.model import HybridRecommender

app=FastAPI(title="Hybrid Recommender System",version="0.1.0")

@lru_cache(maxsize=1)
def engine():
    base=os.getenv("RECOMMENDER_DATA","data/public")
    return HybridRecommender(pd.read_csv(f"{base}/items.csv"),pd.read_csv(f"{base}/interactions.csv"))

@app.get("/health")
def health():return {"status":"ok"}

@app.get("/api/items")
def items():return {"items":engine().items.to_dict(orient="records")}

@app.get("/api/users")
def users():return {"users":sorted(engine().ratings.user_id.unique().tolist())}

@app.get("/api/recommendations/{user_id}")
def recommend(user_id: str,limit:int=Query(5,ge=1,le=30),content_weight:float=Query(.55,ge=0,le=1)):
    try:return {"user_id":user_id,"recommendations":engine().recommend(user_id,limit,content_weight),
                 "note":"Hybrid scores are not calibrated probabilities; artificial CC0 example data"}
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc
