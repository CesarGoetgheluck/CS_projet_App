import streamlit as st
import pandas as pd
import spotipy
from PIL import Image
import plotly.graph_objects as go
from spotipy.oauth2 import SpotifyClientCredentials

# Load logo
logo = Image.open("soundsphere_logo.png")
st.sidebar.image(logo, use_container_width=True)
st.sidebar.markdown("<p style='text-align: center; font-style: italic;'>Your mood. Your music.</p>", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv('tcc_ceds_music.csv').dropna()
    df['release_date'] = df['release_date'].astype(int)
    df['era'] = df['release_date'].apply(lambda y: f"{(y//10)*10}s")
    return df

df = load_data()

# Spotify API setup
client_id = st.secrets["CLIENT_ID"]
client_secret = st.secrets["CLIENT_SECRET"]
sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret
    )
)

# Sidebar content
st.sidebar.subheader("About")
st.sidebar.markdown("This app helps you explore songs by genre, era, and vibe.")
st.sidebar.subheader("Data Source")
st.sidebar.markdown("Data: TCC CEDS Music dataset.")
st.sidebar.subheader("Team")
st.sidebar.markdown("Developed by Team 05.01.")

# Initialize session state
if 'library' not in st.session_state:
    st.session_state['library'] = []

# App title
st.title("SoundSphere, personalised music recommender")
st.markdown("""
1. Choose **one or more genres**, an **era**, and tune audio feature sliders.  
2. Click **generate** to get songs.  
3. Click **regenerate** to refresh.
""")

# Filters
genres = st.multiselect("Genres", df['genre'].unique())
era = st.selectbox("Era", df['era'].unique())

dance_min, dance_max = st.slider("Danceability", 0.0, 1.0, (0.0, 1.0), step=0.01)
energy_min, energy_max = st.slider("Energy", 0.0, 1.0, (0.0, 1.0), step=0.01)
acoustic_min, acoustic_max = st.slider("Acousticness", 0.0, 1.0, (0.0, 1.0), step=0.01)
valence_min, valence_max = st.slider("Valence", 0.0, 1.0, (0.0, 1.0), step=0.01)
loud_min, loud_max = st.slider("Loudness (dB)", float(df['loudness'].min()), float(df['loudness'].max()), step=0.1)
count = st.slider("Number of songs", 1, 5, 1)

# Filter state logic
current = (tuple(genres), era, count)
if st.session_state.get('last_filters') != current:
    st.session_state['last_filters'] = current
    st.session_state.pop('recs', None)
    st.session_state.pop('generated', None)

# Apply filters
filtered = df.copy()
if genres:
    filtered = filtered[filtered['genre'].isin(genres)]
filtered = filtered[filtered['era'] == era]

if filtered.empty:
    st.warning("No songs found for this filter.")
else:
    generated = st.session_state.get('generated', False)
    label = "generate" if not generated else "regenerate"

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

            # Spotify preview
            res = sp.search(q=f"{s['track_name']} {s['artist_name']}", type='track', limit=1)
            if res['tracks']['items']:
                url = res['tracks']['items'][0]['external_urls']['spotify']
                st.markdown(f"[play on Spotify]({url})")

                # Album cover
                album_images = res['tracks']['items'][0]['album']['images']
                if album_images:
                    st.image(album_images[0]['url'], caption='Album cover', width=200)

            # Radar chart
            features = ['danceability', 'energy', 'acousticness', 'valence', 'loudness']
            values = [s[feature] for feature in features]
            values[-1] = (values[-1] - df['loudness'].min()) / (df['loudness'].max() - df['loudness'].min())

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