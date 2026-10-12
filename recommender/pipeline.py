"""Validate and materialize domain-specific Bronze/Silver/Gold CSV datasets."""
import json
from pathlib import Path
import pandas as pd

def run(source="data/public", output="artifacts"):
    base=Path(source); root=Path(output)
    items=pd.read_csv(base/"items.csv")
    ratings=pd.read_csv(base/"interactions.csv")
    expected_items={"item_id","title","category","description"}
    expected_ratings={"user_id","item_id","rating","timestamp"}
    if not expected_items.issubset(items.columns) or not expected_ratings.issubset(ratings.columns):
        raise ValueError("Missing source schema fields")
    for layer in ["bronze","silver","gold"]: (root/layer).mkdir(parents=True,exist_ok=True)
    items.to_csv(root/"bronze"/"items.csv",index=False)
    ratings.to_csv(root/"bronze"/"interactions.csv",index=False)
    valid=ratings["rating"].between(1,5)&ratings["item_id"].isin(items["item_id"])&ratings["user_id"].notna()
    valid &=pd.to_datetime(ratings["timestamp"],errors="coerce").notna()
    valid &=~ratings.duplicated(subset=["user_id","item_id"],keep="last")
    clean=ratings.loc[valid].copy()
    clean.to_csv(root/"silver"/"interactions.csv",index=False)
    ratings.loc[~valid].to_csv(root/"silver"/"rejected.csv",index=False)
    items.drop_duplicates("item_id").to_csv(root/"silver"/"items.csv",index=False)
    gold=clean.groupby("item_id",as_index=False).agg(rating_count=("rating","size"),mean_rating=("rating","mean"))
    gold.to_csv(root/"gold"/"item_popularity.csv",index=False)
    metrics={"items":int(len(items)),"interactions":int(len(ratings)),"valid":int(len(clean)),"rejected":int((~valid).sum())}
    (root/"metrics.json").write_text(json.dumps(metrics,indent=2))
    return metrics

if __name__=="__main__":print(run())
