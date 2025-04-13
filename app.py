import streamlit as st
import pandas as pd

st.title("Find Your Favorite Music")
st.markdown("This app helps you explore new songs based on your music taste.")
st.sidebar.title("Navigation")
st.sidebar.markdown("Use the sidebar to navigate through the app.")
st.sidebar.subheader("About")
st.sidebar.markdown("This app is built using Streamlit and allows users to explore a dataset of songs.")
st.sidebar.subheader("Contact")
st.sidebar.markdown("Highperformers.")

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('tcc_ceds_music.csv')
    return df

df = load_data()

# Preprocessing example: Remove missing data
df_clean = df.dropna()

# Show data preview
st.subheader("Raw Data")
st.write(df_clean)

# Optional: Show basic stats
st.subheader("Data Summary")
st.write(df_clean.describe())

# Optional: Show a pie chart of genres
st.subheader("Genre Distribution")
genre_counts = df_clean['genre'].value_counts()
st.bar_chart(genre_counts)

# Selectbox filter
genre = st.selectbox('Choose a Genre:', df_clean['genre'].unique())
filtered_df = df_clean[df_clean['genre'] == genre]

st.subheader(f"Songs in Genre: {genre}")
st.write(filtered_df)

# Recommend a random song based on the selected genre
if not filtered_df.empty:
    random_song = filtered_df.sample(1)
    st.subheader("Recommended Song")
    st.write(random_song)
    st.markdown(f"Your recommended {genre} song is: {random_song.iloc[0]['track_name']} by {random_song.iloc[0]['artist_name']} 🎵")
    #st.write(f"This is your recommended song from the genre **{genre}**:")
    #st.write(f"{random_song.iloc[0]['track_name']} by {random_song.iloc[0]['artist_name']}")    
else:
    st.subheader("No songs available in the selected genre.")