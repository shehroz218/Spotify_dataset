from re import X
from unicodedata import name
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict
import spotipy
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
import difflib

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.pipeline import Pipeline
import streamlit as st
from PIL import Image
from utils import get_song_data, get_mean_vector, flatten_dict_list, recommend_songs

image = Image.open('D:\Codes\Spotify API/app\Spotify-icon.jpg')
st.image(image,width=400)

st.title("Spotify Recommendation Engine")
st.markdown("This web application recommendds songs based on your favourite song")

CLIENT_ID="26cc86d7297e4c94be7d142145a49eaf"
CLIENT_SECRET="d6faffea02ef44a992203f03e38a4a3f"

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=CLIENT_ID,
                                                           client_secret=CLIENT_SECRET))

DATA_URL="D:\Codes\Spotify API/tracks_features.csv"


@st.cache(persist=True)
def load_data(nrows):
    data=pd.read_csv(DATA_URL, nrows=nrows)
    return data

spotify_data=load_data(100000)
original_data=spotify_data

spotify_data =spotify_data[['name', 'year', 'duration_ms', 'danceability', 'energy',
       'key', 'loudness', 'mode', 'speechiness', 'acousticness',
       'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']]

number_cols = ['year', 'duration_ms', 'danceability', 'energy',
       'key', 'loudness', 'mode', 'speechiness', 'acousticness',
       'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature'] 

song_cluster_pipeline = Pipeline([('scaler', StandardScaler()), 
                                  ('mbk', KMeans(n_clusters=20, 
                                   verbose=2))], verbose=True)


n_in = st.text_input("enter song name" )
y_in = st.number_input("enter year of the song",min_value=1930, max_value=2021)
n_songs =st.number_input('Number of Songs to Recommend', min_value=1, max_value=15)

button=st.button('Recommend Song')

if button:
    x=recommend_songs([{'name':n_in, 'year':y_in}], spotify_data, n_songs=n_songs)
    st.write(x)
    


