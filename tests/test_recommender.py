import pandas as pd
from fastapi.testclient import TestClient
from backend.main import app
from recommender.model import HybridRecommender
from recommender.pipeline import run

def test_etl(tmp_path):
    root=tmp_path/"input";root.mkdir()
    pd.DataFrame([{"item_id":1,"title":"A","category":"a","description":"alpha"},
                  {"item_id":2,"title":"B","category":"b","description":"beta"}]).to_csv(root/"items.csv",index=False)
    pd.DataFrame([{"user_id":"u","item_id":1,"rating":5,"timestamp":"2026-01-01"},
                  {"user_id":"u","item_id":2,"rating":7,"timestamp":"2026-01-01"}]).to_csv(root/"interactions.csv",index=False)
    report=run(str(root),str(tmp_path/"lake"))
    assert report["valid"]==1 and report["rejected"]==1

def test_seen_exclusion_and_cold_start():
    items=pd.read_csv("data/public/items.csv")
    ratings=pd.read_csv("data/public/interactions.csv")
    model=HybridRecommender(items,ratings)
    seen=set(ratings.loc[ratings.user_id=="u1","item_id"])
    recommendations=model.recommend("u1",5)
    assert len(recommendations)==5
    assert not (seen & {x["item_id"] for x in recommendations})
    cold=model.recommend("unknown",3)
    assert len(cold)==3
    assert cold==model.recommend("unknown",3)

def test_api():
    client=TestClient(app)
    assert client.get("/health").status_code==200
    assert len(client.get("/api/items").json()["items"])==12
    assert client.get("/api/recommendations/u1?limit=3").status_code==200
    assert client.get("/api/recommendations/u1?limit=0").status_code==422
