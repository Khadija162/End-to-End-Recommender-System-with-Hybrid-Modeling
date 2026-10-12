"""Streamlit interface for hybrid recommendation experiments."""
import os
import requests
import pandas as pd
import streamlit as st
st.set_page_config(page_title="Hybrid Recommender",layout="wide")
st.title("Hybrid Recommendation Studio")
st.caption("Collaborative filtering + content TF-IDF + cold-start popularity")
api=os.getenv("API_URL","http://localhost:8000")
try:
    response=requests.get(api+"/api/users",timeout=8);response.raise_for_status()
    users=response.json()["users"]
except requests.RequestException as exc:
    users=[]
    st.error(f"Backend unavailable: {exc}")
choice=st.selectbox("User profile",users+["new-user"] if users else ["new-user"])
weight=st.slider("Content modeling weight",0.0,1.0,.55,.05)
count=st.slider("Number of recommendations",1,10,5)
if st.button("Generate recommendations"):
    try:
        r=requests.get(f"{api}/api/recommendations/{choice}",params={"limit":count,"content_weight":weight},timeout=12)
        r.raise_for_status()
        result=r.json()
        frame=pd.DataFrame(result["recommendations"])
        st.dataframe(frame,use_container_width=True)
        if not frame.empty:st.bar_chart(frame.set_index("title")["rank_score"])
        st.info(result["note"])
    except requests.RequestException as exc:st.error(str(exc))
