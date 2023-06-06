import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import numpy

data = pd.read_csv('songs.csv')

st.sidebar.subheader('Selecione a Opçõe:')
show_data = st.sidebar.checkbox('Mostrar dataset')
quantidade = st.sidebar.slider('Escolha a quantidade de artistas', 0, 798)
quantidade_albuns = st.sidebar.slider('Escolha a quantidade de albuns', 0, 798)

if show_data:
    st.subheader('Dados do dataset:')
    st.dataframe(data)

#Mostrando a quantidade total de popularidade
st.header('Quantidade de popularidade')
fig,ax = plt.subplots()
ax.set_xlabel('Nivel de Popularidade')
ax.set_ylabel('Quantidade')
sns.histplot(data, x='Popularity', kde=True, color='g')
st.pyplot(fig)

#Mostrando a quantidade artistas e sua quantidade de música
st.header('Quantidade de Músicas por Artista')
fig2,ax2 = plt.subplots()
qtd_artist = data.loc[:quantidade,'Artist']
ax2.set_xlabel('Artistas')
ax2.set_ylabel('Quantidade de Músicas')
sns.lineplot(x=qtd_artist.values,y=qtd_artist.index, color='b',)
st.pyplot(fig2)

st.header('Quantidade de musicas por album')
fig3,ax3 = plt.subplots()
qtd_alb_art = data.loc[:quantidade_albuns,'Album']
ax3.set_xlabel('Artistas')
ax3.set_ylabel('Quantidade de Albuns')
sns.lineplot(y=qtd_alb_art.index, x=qtd_alb_art.values, color='r',)
st.pyplot(fig3)

data['Overview'] = data['Artist'] + ". " + data['Album'] + ". " + data['Popularity'].astype(str) + ". " + data['Lyrics']
data['Overview'] = data['Overview'].apply(lambda x: x.lower())
data['Name'] = data['Name'].apply(lambda x: x.lower())

cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(data['Overview']).toarray()
similarity = cosine_similarity(vector)

def similar_song(name):
    texto = ""
    aux = 0
    name = name.lower()
    indices = data[data['Name'] == name].index[0]
    distances = similarity[indices]
    arr = sorted(list(enumerate(distances)), reverse = True, key=lambda x: x[1])[1:6]
    print("Recommended options are:")
    for i in arr:
        song_name = data.loc[i[0], 'Name']
        artist = data.loc[i[0], 'Artist']
        album = data.loc[i[0],'Album']
        texto += song_name.capitalize() + " " + 'by' + " " + artist + " " + 'album' + " " + '-' + " " + album + "\n"
        aux += 1
    return texto

texto = similar_song("Imagine - Remastered 2010")
st.subheader('Recomendações para a música: Imagine - Remastered 2010 ')
st.text(texto)