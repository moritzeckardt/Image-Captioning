import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
import pickle 
import random
from urllib.request import urlopen
import requests
import streamlit as st


#load data

features = pickle.load(open('./images1.pkl', 'rb'))

words_to_index = pickle.load(open('words.pkl', 'rb'))
index_to_words = pickle.load(open('words1.pkl', 'rb'))

model = load_model('model_19.h5')

images = "Images/"


max_length = 43


#generate captions
def image_captioning(picture):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [words_to_index[w] for w in in_text.split() if w in words_to_index]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([picture,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = index_to_words[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

def org_caption(id):
    original = 0
    with open("captions.txt", 'r', encoding='utf-8') as f:
        caps = [line.rstrip() for line in f]
        for i in caps:
            if id == i:
                print(i)

    return original


#show picture an caption
def show():
    z = random.randint(0,500)
    pic = list(features.keys())[z]
    image = features[pic].reshape((1,2048))
    x = plt.imread(images + pic)
    st.image(x)
    st.write("Generated caption:", image_captioning(image))
    st.write("Original caption:, ", org_caption(pic))
    
if st.button("Generate caption"):
    show()
 