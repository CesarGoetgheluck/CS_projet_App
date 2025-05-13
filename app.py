import streamlit as st
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sqlite3
from PIL import Image

# --- UI Theming ---
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #cce6ff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load logo
logo = Image.open("soundsphere_logo.png")
st.sidebar.image(logo)  # Removed use_container_width argument
st.sidebar.markdown("<p style='text-align: center; font-style: italic;'>Your mood. Your music.</p>", unsafe_allow_html=True)

# --- Database Initialization ---
def init_db():
    conn = sqlite3.connect('music.db')
    c = conn.cursor()
    # Playlists tables
    c.execute('''
        CREATE TABLE IF NOT EXISTS Playlists (
            playlist_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       INTEGER,
            playlist_name TEXT,
            description   TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS Playlist_Songs (
            playlist_id  INTEGER,
            track_name   TEXT,
            artist_name  TEXT,
            genre        TEXT,
            topic        TEXT,
            release_date INTEGER,
            FOREIGN KEY(playlist_id) REFERENCES Playlists(playlist_id)
        )
    ''')
    conn.commit()
    conn.close()

# Initialize database tables once
init_db()

# --- Data Loading from SQLite ---
@st.cache_data


import os

def load_data():
    if not os.path.exists("music.db"):
        df = pd.read_csv("tcc_ceds_music.csv").dropna()
        df['release_date'] = df['release_date'].astype(int)
        df['era'] = df['release_date'].apply(lambda y: f"{(y // 10) * 10}s")
        conn = sqlite3.connect("music.db")
        df.to_sql("songs", conn, if_exists="replace", index=False)
        conn.close()

    conn = sqlite3.connect("music.db")
    df = pd.read_sql_query("SELECT * FROM songs", conn)
    return df


df = load_data()
if 'library' not in st.session_state:
    st.session_state['library'] = []

# --- Spotify API Setup ---
client_id = st.secrets["CLIENT_ID"]
client_secret = st.secrets["CLIENT_SECRET"]
sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(client_id, client_secret)
)

@st.cache_data
def get_spotify_url(query):
    res = sp.search(q=query, type='track', limit=1)
    if res['tracks']['items']:
        return res['tracks']['items'][0]['external_urls']['spotify']
    return None

# --- Machine Learning Model ---
le = LabelEncoder()
df['genre_label'] = le.fit_transform(df['genre'])
features = ['danceability', 'energy', 'acousticness', 'valence', 'loudness', 'instrumentalness']
X = df[features]
y = df['genre_label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
genre_clf = DecisionTreeClassifier(max_depth=5)
genre_clf.fit(X_train, y_train)
importance = genre_clf.feature_importances_

# --- Playlist Helper Functions ---
def create_playlist(user_id, name, desc):
    conn = sqlite3.connect('music.db')
    c = conn.cursor()
    c.execute(
        "INSERT INTO Playlists (user_id, playlist_name, description) VALUES (?, ?, ?)",
        (user_id, name, desc)
    )
    conn.commit()
    conn.close()
    st.success(f"Playlist '{name}' created!")


def add_song_to_playlist(playlist_id, song):
    conn = sqlite3.connect('music.db')
    c = conn.cursor()
    c.execute(
        "SELECT 1 FROM Playlist_Songs WHERE playlist_id=? AND track_name=? AND artist_name=?",
        (playlist_id, song['track_name'], song['artist_name'])
    )
    if c.fetchone():
        st.warning("Song already in playlist.")
    else:
        c.execute(
            '''INSERT INTO Playlist_Songs 
               (playlist_id, track_name, artist_name, genre, topic, release_date)
               VALUES (?, ?, ?, ?, ?, ?)''',
            (
                playlist_id,
                song['track_name'], song['artist_name'],
                song['genre'], song['topic'], song['release_date']
            )
        )
        conn.commit()
        st.success("Added to playlist!")
    conn.close()


def get_user_playlists(user_id):
    conn = sqlite3.connect('music.db')
    c = conn.cursor()
    c.execute(
        "SELECT playlist_id, playlist_name, description FROM Playlists WHERE user_id=?",
        (user_id,)
    )
    pls = c.fetchall()
    conn.close()
    return pls


def get_playlist_songs(playlist_id):
    conn = sqlite3.connect('music.db')
    c = conn.cursor()
    c.execute(
        "SELECT track_name, artist_name FROM Playlist_Songs WHERE playlist_id=?",
        (playlist_id,)
    )
    songs = c.fetchall()
    conn.close()
    return songs

# --- Streamlit Tabs ---
tab1, tab2, tab3 = st.tabs([
    "🎵 Music Recommender",
    "🎶 Genre Predictor",
    "📀 Playlists"
])

# --- Tab 1: Music Recommender ---
with tab1:
    st.title("🎵 Personalised Music Recommender")
    st.markdown("""
    1. Choose **genres**, **era**, **topic** (optional), **audio features** and **number of songs**.
    2. Click **generate** to get songs.
    3. Add favorites to your **personal library**.
    """)

    genres = st.multiselect("Genres", df['genre'].unique())
    era = st.selectbox("Era", df['era'].unique())
    topics = st.multiselect("Topics", df['topic'].unique())
    dance_min, dance_max = st.slider("Danceability", 0.0, 1.0, (0.0, 1.0), 0.01)
    energy_min, energy_max = st.slider("Energy", 0.0, 1.0, (0.0, 1.0), 0.01)
    acoustic_min, acoustic_max = st.slider("Acousticness", 0.0, 1.0, (0.0, 1.0), 0.01)
    valence_min, valence_max = st.slider(
    "Valence",
    0.0, 1.0, (0.0, 1.0),
    step=0.01,
    help="Valence measures musical positiveness: high (near 1) means cheerful/upbeat, low (near 0) means sad/tense"
)
    loud_min, loud_max = st.slider(
        "Loudness (dB)", float(df['loudness'].min()), float(df['loudness'].max()),
        (float(df['loudness'].min()), float(df['loudness'].max())), 0.1
    )
    count = st.slider("Number of songs", 1, 5, 1)

    current = (tuple(genres), era, tuple(topics), count)
    if st.session_state.get('last_filters') != current:
        st.session_state['last_filters'] = current
        st.session_state.pop('recs', None)
        st.session_state.pop('generated', None)

    filtered = df.copy()
    if genres:
        filtered = filtered[filtered['genre'].isin(genres)]
    if topics:
        filtered = filtered[filtered['topic'].isin(topics)]
    filtered = filtered[(filtered['era'] == era) &
                        (filtered['danceability'].between(dance_min, dance_max)) &
                        (filtered['energy'].between(energy_min, energy_max)) &
                        (filtered['acousticness'].between(acoustic_min, acoustic_max)) &
                        (filtered['valence'].between(valence_min, valence_max)) &
                        (filtered['loudness'].between(loud_min, loud_max))]

    if filtered.empty:
        st.warning("No songs found.")
    else:
        label = "generate" if not st.session_state.get('generated') else "regenerate"
        def generate_songs():
            n = min(count, len(filtered))
            recs = filtered.sample(n).sort_values('release_date', ascending=False)
            st.session_state['recs'] = recs
            st.session_state['generated'] = True
        st.button(label, on_click=generate_songs)

    recs = st.session_state.get('recs')
    if recs is not None:
        for i, (_, s) in enumerate(recs.iterrows()):
            st.markdown(f"**{s['track_name']}** by {s['artist_name']} ({s['release_date']})")
            url = get_spotify_url(f"{s['track_name']} {s['artist_name']}")
            if url:
                st.markdown(f"[Play on Spotify]({url})")

                # Album cover
                res = sp.search(q=f"{s['track_name']} {s['artist_name']}", type='track', limit=1)
                if res['tracks']['items']:
                    album_images = res['tracks']['items'][0]['album']['images']
                    if album_images:
                        st.image(album_images[0]['url'], caption='Album cover', width=200)
                else:
                    st.warning("Album cover not found.")

            # Radar chart
            features_chart = ['danceability', 'energy', 'acousticness', 'valence']
            values = [s[f] for f in features_chart]
            loud_norm = (s['loudness'] - loud_min) / (loud_max - loud_min)
            values.append(min(max(loud_norm, 0), 1))

            fig = go.Figure(
                data=[go.Scatterpolar(
                    r=values + [values[0]],
                    theta=features_chart + ['loudness', features_chart[0]],
                    fill='toself'
                )]
            )
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                showlegend=False,
                title="Audio Feature Profile"
            )
            st.plotly_chart(fig, use_container_width=True)

            def add_to_library(track=s['track_name'], artist=s['artist_name'], genre=s['genre'], topic=s['topic'], release=s['release_date']):
                if not any(item['track_name'] == track and item['artist_name'] == artist for item in st.session_state['library']):
                    st.session_state['library'].append({'track_name': track, 'artist_name': artist, 'genre': genre, 'topic': topic, 'release_date': release})
                    # Fixed the error by removing the conditional usage of st.toast
                    st.success(f"Added '{track}' to your library!")
                else:
                    st.warning("This song is already in your library.")
            st.button("Add to library", key=f"like_{i}", on_click=add_to_library)

    if st.session_state['library']:
        st.subheader("Your library")
        lib_df = pd.DataFrame(st.session_state['library'])
        sort_by = st.selectbox("Sort library by", lib_df.columns)
        st.table(lib_df.sort_values(sort_by))
        csv = lib_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download library as CSV", data=csv, file_name='my_library.csv', mime='text/csv')

# --- Tab 2: Genre Predictor ---
with tab2:
    st.title("🎶 Genre Predictor")
    st.markdown("What is my best fitting genre? Find out here by adding your audio feautures to get your predicted genre with probability breakdown and importance of the feautures.")

    dance = st.slider("Danceability", 0.0, 1.0, 0.5)
    energy = st.slider("Energy", 0.0, 1.0, 0.5)
    acoustic = st.slider("Acousticness", 0.0, 1.0, 0.5)
    valence = st.slider("Valence", 0.0, 1.0, 0.5)
    loud = st.slider("Loudness (dB)", float(df['loudness'].min()), float(df['loudness'].max()), 0.0)
    instr = st.slider("Instrumentalness", 0.0, 1.0, 0.5)

    input_features = pd.DataFrame([[dance, energy, acoustic, valence, loud, instr]], columns=features)

    if st.button("Predict Genre"):
        probs = genre_clf.predict_proba(input_features)[0]
        genres_pred = le.inverse_transform(range(len(probs)))
        prob_df = pd.DataFrame({'Genre': genres_pred, 'Probability': probs}).sort_values('Probability', ascending=False)

        top_genre = prob_df.iloc[0]['Genre']
        st.success(f"🎧 Suggested Genre: **{top_genre}**")

        st.subheader("Probability Breakdown")
        fig = go.Figure(go.Bar(x=prob_df['Probability'], y=prob_df['Genre'], orientation='h'))
        fig.update_layout(xaxis_title="Probability", yaxis_title="Genre")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Feature Importance")
        importance_df = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values('Importance', ascending=True)
        fig_imp = go.Figure(go.Bar(x=importance_df['Importance'], y=importance_df['Feature'], orientation='h'))
        fig_imp.update_layout(xaxis_title="Importance", yaxis_title="Feature")
        st.plotly_chart(fig_imp, use_container_width=True)

# --- Tab 3: Playlists ---
with tab3:
    st.title("📀 Your Playlists")
    st.markdown("Create and manage your personalised playlists here.")
    user_id = 1  # placeholder until auth implemented

    with st.form("create_playlist_form"):
        name = st.text_input("Playlist Name")
        desc = st.text_area("Description")
        if st.form_submit_button("Create Playlist"):
            create_playlist(user_id, name, desc)

    playlists = get_user_playlists(user_id)
    if playlists:
        sel_labels = {f"{p[1]} (#{p[0]})": p[0] for p in playlists}
        sel = st.selectbox("Select Playlist to Add Recs", list(sel_labels.keys()))
        pid = sel_labels[sel]
        recs = st.session_state.get('recs')
        if recs is not None:
            for _, song in recs.iterrows():
                st.write(f"{song['track_name']} by {song['artist_name']} ({song['release_date']})")
                if st.button(f"➕ Add to {sel}", key=f"add_{song['track_name']}"):
                    add_song_to_playlist(pid, song)
    else:
        st.info("No playlists yet. Create one above!")

    st.markdown("---")
    for pid, pname, pdesc in playlists:
        st.subheader(pname)
        if pdesc:
            st.caption(pdesc)
        songs = get_playlist_songs(pid)
        if songs:
            for t, a in songs:
                st.write(f"- **{t}** by {a}")
        else:
            st.write("_Playlist is empty._")

            
            
