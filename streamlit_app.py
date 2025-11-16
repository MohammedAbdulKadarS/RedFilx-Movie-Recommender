import streamlit as st
import pandas as pd
import ast
import joblib

st.set_page_config(page_title="🔥 Movie Recommender", page_icon="🎦", layout="centered")

# ---- Custom Styles ----
st.markdown("""
    <style>
        body, .stApp {
            background: #111112 !important;
            color: #fff !important;
        }
        h1,h2,h3,h4 {
            color:#f52331 !important;
            font-family: 'Segoe UI',sans-serif;
            font-weight: 700;
        }
        label, .stSlider label, .stSelectbox label, .stTextInput label {
            color: #fff !important;
            font-size: 16px;
            font-family: 'Segoe UI',sans-serif;
        }
        .stButton>button {
            background: linear-gradient(90deg, #f52331 60%, #a00a15 100%);
            color: #fff;
            border-radius: 12px;
            font-size: 18px;
            font-weight: bold;
            box-shadow: 0 0 22px #f5233144;
        }
        .stButton>button:hover {
            background: #f52331;
            box-shadow: 0 0 30px #f52331;
            color:#fff;
        }
        .card-glow {
            border: 2px solid #f52331;
            border-radius: 18px;
            box-shadow: 0 0 25px #f5233144;
            padding: 1.1em;
            margin-bottom: 1.3em;
            background:#1c1c1f;
            animation: fade-in 0.8s;
        }
        .card-glow .movie-title { color: #f52331; font-size:23px; font-family:'Segoe UI',sans-serif;font-weight:700;}
        .card-glow .result-text { color: #fff; font-size:16px;}
        .card-glow .movie-info { color: #fff; font-size:14px;}
        @keyframes fade-in {
            from { opacity: 0; transform: translateY(30px);}
            to { opacity: 1; transform: translateY(0);}
        }
    </style>
""", unsafe_allow_html=True)

st.title("🔥 Next-Gen Movie Recommender App")
st.markdown("""<span style='color:#f52331;font-size:28px;font-weight:700;'>Pick a Genre (required) & Suggest a watched film (optional). Get smart recommendations! 🚀</span>""", unsafe_allow_html=True)

# ---- Data Load & Prep ----
df = pd.read_csv(r'C:\Users\abdula\OneDrive\Desktop\MovieRecomendationKNN\movie_prediction_dataset.csv')
df['overview'] = df['overview'].fillna('')
df['genres'] = df['genres'].fillna('[]')
df['genres'] = df['genres'].apply(lambda x: ' '.join([i['name'] for i in ast.literal_eval(x)]))
df['combined_features'] = df['overview'] + ' ' + df['genres']
tfidf = joblib.load('tfidf_vectorizer.joblib')
knn = joblib.load('knn_model.joblib')

# ---- UI ----
genre_options = sorted(set(df['genres'].str.split().sum()))
chosen_genre = st.selectbox("🎨 Your favourite genre (must select):", genre_options)
movies_in_genre = df[df['genres'].str.contains(chosen_genre)]['title'].sort_values().unique()
watched_movie = st.selectbox("🎬 [Optional] Movie you've watched in this genre:", ["None"] + list(movies_in_genre))
n_recs = st.slider("How many recommendations?", 3, 10, 5)
st.divider()

# ---- Logic ----
def recommend_for_genre(genre, top_n=5):
    genre_df = df[df['genres'].str.contains(genre)].copy()
    genre_df = genre_df.sort_values("popularity", ascending=False)
    return genre_df.head(top_n)[['title', 'genres', 'overview']]

def recommend_for_genre_and_movie(genre, movie_title, top_n=5):
    idx = df[df['title'] == movie_title].index[0]
    features = tfidf.transform([df.iloc[idx]['combined_features']])
    distances, indices = knn.kneighbors(features)
    similar_titles = [df.iloc[i]['title'] for i in indices[0] if df.iloc[i]['title']!=movie_title]
    final_recs = []
    for rec in similar_titles:
        if genre in df[df['title']==rec].iloc[0]['genres']:
            final_recs.append(rec)
        if len(final_recs) >= top_n:
            break
    return [df[df['title']==t][['title','genres','overview']].iloc[0] for t in final_recs]

if st.button("🔍 Find Movies!"):
    with st.spinner('Finding epic films just for you... 🍿'):
        if watched_movie != "None":
            st.success(f"Movies in '{chosen_genre}' similar to '{watched_movie}':")
            recs = recommend_for_genre_and_movie(chosen_genre, watched_movie, n_recs)
        else:
            st.success(f"Best movies in '{chosen_genre}':")
            recs = recommend_for_genre(chosen_genre, n_recs).values.tolist()
        for i, item in enumerate(recs, 1):
            title, genres, overview = item[0], item[1], item[2]
            st.markdown(f"""
                <div class='card-glow'>
                    <span class='movie-title'>{i}. {title}</span><br>
                    <span class='result-text'><b>Genres:</b> {genres}</span><br>
                    <span class='result-text'><b>Overview:</b> <span class='movie-info'>{overview[:240]}...</span></span>
                </div>
            """, unsafe_allow_html=True)

st.markdown("<hr style='border:1.5px solid #f52331;'>", unsafe_allow_html=True)
st.markdown("""
<hr style="border:2px solid #f52331;margin-top:2em;">
<div style="color:#fff; font-size:19px; font-family:'Segoe UI',sans-serif;text-align:center; margin-top:7px;">
    🚀 <span style="color:#f52331; font-weight:700;">RedFlix Movie Recommender</span> | Fast, Smart, and Stylish. <br>
    <span style="color:#fff;font-weight:400;font-size:15px;">Discover your next favourite film with AI-powered recommendations.<br>
    Built for professionals, students, and movie lovers. <br>
    Try. Enjoy. Share. 🎬</span>
</div>
""", unsafe_allow_html=True)

