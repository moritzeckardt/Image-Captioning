from pyexpat import model
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
import pickle
import streamlit as st
import pandas as pd
from model_test import show, image_captioning, org_caption




# Headline
st.write("# ML4B - Image Caption Generator")

# Project explanation
st.header("Short project explanation")
st.write("""Hello, we are Jonas, Moritz and Ole. Together we have set ourselves the goal of building, explaining and 
presenting an image caption generator using Deep Learning and its Neural Networks. We use these subtype of machine 
learning, as it is the closest to the way humans analyze images. Based on this, we want to compare our own generated 
caption with the actual caption of the image. The tip of the iceberg would be if we manage to algorithmically 
evaluate the appropriateness of our caption.""")
st.write("")

# Data preparation
st.header("Our data preparation process")

with st.expander("1. Data Understanding"):
    st.write("""For our project we are using the LAION data set, which is available to us as a CSV file. It is currently 
    the largest freely accessible image-text dataset in the world (240TB). The CSV file contains various attributes, 
    for example URL, TEXT, NSFW or Similarity. The URL can be used to load numerous images and display them. 
    The data set sets the ground for our goal of generating image captions. Of course it is too large in its actual 
    form, which is why we decided to use a smaller sample. We agreed on a sample size of about 1,000 - 5,000 examples. 
    At the moment, we have focused on the following 3 image themes when training the model: Cars, Chairs and Couches. 
    One of the problems with our project could be the comparison of the captions, because some of the captions are very 
    specific and do not always reflect the actual content of the images.""")

with st.expander("2. Source Selection"):
    st.write("""As already explained in the previous step, we make use of the LAION data set. The total data for 
    training, validating and testing will initially be 1,000 - 5,000 images to not make the process too time-consuming.
    We are not concerned about missing data sources with this data set.""")

with st.expander("3. Data Cleaning"):
    st.write("""Corrupted data cannot be found in our data set. In rare cases, an image cannot be be loaded. However, 
    in our opinion there is nothing to be done about this problem. We would like to clean our images for english 
    captions, NSFW, a maximum caption length as well as appropriate keywords.""")

with st.expander("4. Feature Engineering"):
    st.write("""In terms of feature engineering, we might use the typical techniques for images:
     Resizing, cropping, clipping, blur, etc. We keep our options open in this area and analyze what could be 
     useful.""")

with st.expander("5. Data Splitting"):
    st.write("""For the time being, we would like to start with a total of about 5,000 images. These will be divided 
    into a Train Set (70-80%), Validation Set (10-15%) and Test Set (10-15%). The process of dividing will be 
    random.""")
st.write("")


# Load data
@st.cache
def load_data():
    dataframe = pd.read_csv("")
    return dataframe


data_load_state = st.header("Loading data...")

#df = load_data()

data_load_state.header("Generated caption vs. real caption:")


# Get image
def get_image():
    column = random.randint(0, len(df) - 1)
    url = df.iloc[column]["url"]
    st.image(url)
    st.write("Real caption: ")
    st.write(df.iloc[column]["caption"])
    st.write("URL:")
    st.write(url)


#center the button

col1, col2, col3 = st.columns([1,1,1])
#Generate caption to corresponding image
with col1:
    pass
with col2:
    if st.button("Generate caption:"):
        show()





