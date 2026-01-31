import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Hotel Recommender", layout="wide")

with open("hotel.pkl", "rb") as f:
    data = pickle.load(f)

df = data["df"]

import base64

def get_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_image = get_base64("assets/bg2.jpg")


# ---------------- CSS ----------------
st.markdown(f"""
<style>
[data-testid="stApp"] {{
    background:
    linear-gradient(rgba(2,6,23,0.65), rgba(2,6,23,0.85)),
    url("data:image/jpg;base64,{bg_image}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    font-family: 'Segoe UI', sans-serif;
}}

.hero {{
    max-width:720px;
    margin:auto;
    padding:32px;
    background: linear-gradient(145deg,#020617,#020617);
    border-radius:22px;
    text-align:center;
    box-shadow:0 0 25px rgba(37,99,235,0.25);
}}

.hero h1 {{
    color:#3B82F6;
    font-size:40px;
    font-weight:800;
}}

.hero p {{
    color:#CBD5F5;
}}

.grid {{
    display:grid;
    grid-template-columns: repeat(3,1fr);
    gap:22px;
}}

.hotel-card {{
    background: linear-gradient(145deg,#020617,#020617);
    border:1px solid #1E293B;
    border-radius:18px;
    padding:18px;
    color:#E5E7EB;
    box-shadow:0 0 18px rgba(37,99,235,0.15);
    transition:0.25s;
}}

.hotel-card:hover {{
    transform:scale(1.03);
    box-shadow:0 0 28px rgba(37,99,235,0.35);
    border:1px solid #3B82F6;
}}

.book-btn {{
    color:#BFDBFE;
    font-weight:600;
    text-decoration:none;
}}
.book-btn:hover {{
    color:#DBEAFE;
}}
</style>
""", unsafe_allow_html=True)



# ---------------- Recommendation Logic ----------------


def recommend(hotel,n=6):
    vect = data['dict'][hotel]
    sim = cosine_similarity(vect,data['x'])[0]
    idx = sim.argsort()[::-1][1:n+1]
    return df['hotelname'].iloc[idx].tolist()



# ---------------- Hero ----------------
st.markdown("""
<div class="hero">
    <h1>🏨 Hotel Recommendation System</h1>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

center = st.columns([1,2,1])
with center[1]:
    selected = st.selectbox("Choose a Hotel", list(data['dict'].keys()))
    btn = st.button("Find Hotels")



# ---------------- Hotel Card ----------------
def card(name):
    h = df[df.hotelname==name].iloc[0]
    return f"""
    <div class="hotel-card">
        <h3>{h['hotelname']}</h3>
        <p>📍 {h['address']}</p>
        <p>🌆 {h['city']} , {h['country']}</p>
        <p>⭐ Star Rating: {h['starrating']}</p>
        <a class="book-btn" href="{h['url']}" target="_blank">🔗 Book Now</a>
    </div>
    """

# ---------------- Output ----------------
if btn:
    st.subheader("Selected Hotel")
    st.markdown(card(selected), unsafe_allow_html=True)

    st.subheader("Recommended Hotels")

    recs = recommend(selected)

    # First row
    row1 = st.columns(3)
    for col, h in zip(row1, recs[:3]):
        with col:
            st.markdown(card(h), unsafe_allow_html=True)

    # Second row
    row2 = st.columns(3)
    for col, h in zip(row2, recs[3:6]):
        with col:
            st.markdown(card(h), unsafe_allow_html=True)
