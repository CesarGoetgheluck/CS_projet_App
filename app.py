
import streamlit as st
import pandas as pd
#spotipy is a lightweight Python library for forms and charts
import spotipy
#this followig is to import the pnj of the logo
from PIL import Image
logo = Image.open("soundsphere_logo.png")
st.sidebar.image(logo, use_container_width=True)
from spotipy.oauth2 import SpotifyClientCredentials

@st.cache_data
def load_data():
    df = pd.read_csv('tcc_ceds_music.csv').dropna()
    df['release_date'] = df['release_date'].astype(int)
    df['era'] = df['release_date'].apply(lambda y: f"{(y//10)*10}s")
    return df

# load data
df = load_data()

# spotify client setup
client_id = st.secrets["CLIENT_ID"]
client_secret = st.secrets["CLIENT_SECRET"]
sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret
    )
)

# set up sidebar
st.sidebar.subheader("About")
st.sidebar.markdown("This app is built using Streamlit and allows users to explore a dataset of songs. Users can filter songs by genre, era, and topic, and even add their favorite songs to a personal library.")
st.sidebar.subheader("Data Source")
st.sidebar.markdown("The dataset used in this app is from the TCC CEDS Music dataset.")
st.sidebar.subheader("Team")
st.sidebar.markdown("this app is developed by Team 05.01.")

# initialize library
if 'library' not in st.session_state:
    st.session_state['library'] = []

# app ui
title = "SoundSphere " \
"personalised music recommender"
st.title(title)
st.markdown(
    """
1. choose **one or more genres**, **era**, **audio features** and **number of songs**.
2. click **generate** to get songs.
3. click **regenerate** to refresh.
"""
)

# filters
genres = st.multiselect("genres", df['genre'].unique())  # choose one or more genres
era = st.selectbox("era", df['era'].unique())      # choose an era
# audio feature sliders
dance_min, dance_max = st.slider("danceability range", 0.0, 1.0, (0.0, 1.0), step=0.01)
energy_min, energy_max = st.slider("energy range", 0.0, 1.0, (0.0, 1.0), step=0.01)
acoustic_min, acoustic_max = st.slider("acousticness range", 0.0, 1.0, (0.0, 1.0), step=0.01)
valence_min, valence_max = st.slider(
    "valence range",
    0.0, 1.0, (0.0, 1.0),
    step=0.01,
    help="valence measures musical positiveness: high (near 1) means cheerful/upbeat, low (near 0) means sad/tense"
)
loud_min, loud_max = st.slider(
    "loudness (dB)",
    float(df['loudness'].min()),
    float(df['loudness'].max()),
    (float(df['loudness'].min()), float(df['loudness'].max())),
    step=0.1
)
count = st.slider("number of songs", 1, 5, 1) # how many songs to recommend("number of songs", 1, 5, 1)

# clear state on filter change
current = (tuple(genres), era, count)
if st.session_state.get('last_filters') != current:
    st.session_state['last_filters'] = current
    st.session_state.pop('recs', None)
    st.session_state.pop('generated', None)

# filter dataset
filtered = df.copy()
if genres:
    filtered = filtered[filtered['genre'].isin(genres)]  # filter by selected genres
filtered = filtered[filtered['era'] == era]              # filter by era

if filtered.empty:
    st.warning("no songs found.")
else:
    # generate vs. regenerate buttons
    generated = st.session_state.get('generated', False)  # tracks if recommendations generated
    label = "generate" if not generated else "regenerate"

    def generate_songs():
        n = min(count, len(filtered))  # number of songs to sample
        recs = (
            filtered
            .sample(n)  # randomly pick n unique songs
            .sort_values('release_date', ascending=False)  # sort newest first
        )
        st.session_state['recs'] = recs        # store sampled songs
        st.session_state['generated'] = True   # mark generated flag

    st.button(label, on_click=generate_songs)  # clicking invokes call back generate_songs and updates session_state, so that the button label changes to "regenerate recommendations" after the first click

    # display recommendations with like button
    recs = st.session_state.get('recs')
    if recs is not None:
        for i, (_, s) in enumerate(recs.iterrows()):
            st.markdown(f"**{s['track_name']}** by {s['artist_name']} ({s['release_date']})")
            res = sp.search(q=f"{s['track_name']} {s['artist_name']}", type='track', limit=1)
            if res['tracks']['items']:
                url = res['tracks']['items'][0]['external_urls']['spotify']
                st.markdown(f"[play on spotify]({url})")

                # 📈 Radar chart for audio features (INDENTED CORRECTLY)
                import plotly.graph_objects as go

                features = ['danceability', 'energy', 'acousticness', 'valence', 'loudness']
                values = [s[feature] for feature in features]

                # Normalize loudness
                loud_min, loud_max = df['loudness'].min(), df['loudness'].max()
                values[-1] = (values[-1] - loud_min) / (loud_max - loud_min)

                fig = go.Figure(
                    data=[
                        go.Scatterpolar(
                            r=values + [values[0]],
                            theta=features + [features[0]],
                            fill='toself',
                            name=s['track_name']
                        )
                    ]
                )
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=False,
                    title="Audio Feature Profile"
                )

                st.plotly_chart(fig, use_container_width=True)
