import streamlit as st
import random
import pickle 
from model import predict_step

#preloading the images folder
features = pickle.load(open('./images1.pkl', 'rb'))
images = "Images/"

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


# show the picture and generate the caption
def gen_caption(picture):
    st.image(picture)
    st.subheader('Generated caption:')
    with st.spinner(text='This may take a moment...'):
        caption = predict_step([picture])
    st.write(caption[0])


# user chooses between preuploaded picture or uploads one himself
col1, col2, col3 = st.columns([0.1, 1, 0.1])
with col1:
    pass
with col2:
    st.subheader('Caption Generator')
    user_choice = st.radio(label='Choose from either option',
                           options=['Upload your own picture', 'Image from our dataset'])
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

if user_choice == 'Upload your own picture':
    picture = st.file_uploader('', type=['jpg'])
    if picture != None:
        with col1:
            pass
        with col2:
            gen_caption(picture)
else:
    # center the button
    with col1:
        pass
    with col2:
        # get a random picture from out dataset
        if st.button("Get a picture from our dataset:"):
            z = random.randint(0,500)
            picture = list(features.keys())[z]
            picture = features[picture].reshape((1,2048))
            picture = plt.imread(images + picture)
            gen_caption(picture)
