"""Content TF-IDF + user-user collaborative signal + Bayesian popularity fallback."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class HybridRecommender:
    def __init__(self, items: pd.DataFrame, ratings: pd.DataFrame):
        self.items=items.drop_duplicates("item_id").reset_index(drop=True).copy()
        self.ratings=ratings.copy()
        self.ratings=self.ratings[self.ratings.item_id.isin(self.items.item_id) & self.ratings.rating.between(1,5)]
        docs=(self.items["title"].fillna("")+" "+self.items["category"].fillna("")+" "+self.items["description"].fillna(""))
        self.tfidf=TfidfVectorizer(stop_words="english").fit_transform(docs)
        self.users=self.ratings.pivot_table(index="user_id",columns="item_id",values="rating",aggfunc="mean")
        mean=self.ratings.groupby("item_id").rating.agg(["mean","count"])
        prior=float(self.ratings.rating.mean()) if len(self.ratings) else 3.
        self.popularity={int(i):float((r["mean"]*r["count"]+prior*3)/(r["count"]+3)) for i,r in mean.iterrows()}
        self.popularity_prior=prior

    def recommend(self, user_id: str, limit: int=5, content_weight: float=.55):
        if limit<1 or limit>30 or not 0<=content_weight<=1: raise ValueError("Invalid recommendation settings")
        history=self.ratings[self.ratings.user_id==user_id]
        seen=set(history.item_id.astype(int))
        candidate_ids=self.items.item_id.astype(int).tolist()
        content=np.zeros(len(candidate_ids))
        collaborative=np.zeros(len(candidate_ids))
        if len(history):
            weights=history.rating.to_numpy(dtype=float)-3.
            positions=[candidate_ids.index(int(i)) for i in history.item_id if int(i) in candidate_ids]
            weights=weights[:len(positions)]
            if positions and np.abs(weights).sum():
                sims=cosine_similarity(self.tfidf,self.tfidf[positions])
                content=np.asarray(sims@weights / np.abs(weights).sum()).ravel()
            if user_id in self.users.index and len(self.users.index)>1:
                target=self.users.loc[user_id]
                peers=self.users.drop(index=user_id)
                overlap=peers.notna() & target.notna()
                common=overlap.sum(axis=1)
                # Similarity only for actual common-rated items: no zero-imputation.
                differences=peers.sub(target,axis="columns").where(overlap)
                similarity=(1-(differences.abs().sum(axis=1)/common.replace(0,np.nan)/4)).fillna(0)
                similarity=similarity.where(common>=1,0)
                for j,item in enumerate(candidate_ids):
                    available=peers[item].notna() if item in peers else pd.Series(False,index=peers.index)
                    w=similarity.loc[available]
                    if w.sum()>0: collaborative[j]=float((peers.loc[available,item].sub(3)@w)/(2*w.sum()))
        popularity=np.array([self.popularity.get(i,self.popularity_prior)/5 for i in candidate_ids])
        # Scores are illustrative ranking signals, not calibrated probabilities.
        if len(history):
            signal=content_weight*content+(1-content_weight)*collaborative
            scores=.75*signal+.25*popularity
        else: scores=popularity
        ranked=sorted(((candidate_ids[i],float(scores[i])) for i in range(len(candidate_ids)) if candidate_ids[i] not in seen),
                      key=lambda t:(-t[1],t[0]))[:limit]
        lookup=self.items.set_index("item_id")
        return [{"item_id":int(i),"title":str(lookup.loc[i,"title"]),"category":str(lookup.loc[i,"category"]),
                 "rank_score":round(s,4)} for i,s in ranked]
