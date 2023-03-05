import streamlit as st
import random
import pickle 
from model import predict_step
# Configure page 
st.set_page_config(
    page_title = "Image Captioning",
    page_icon = "🖼️"
)

# Preloading the images folder
@st.cache
def load_feature_file():
    loaded_features = pickle.load(open('./images1.pkl', 'rb'))
    return loaded_features
features = load_feature_file()
images = "Images/"

# Headline
st.write("# ML4B - Image Caption Generator")

# Project explanation
st.header("Short project explanation")
st.write("""Hello, we are Jonas, Moritz and Ole. Together we want to understand, build and present a caption generator 
using neural networks. We use this subtype of machine learning because it is closest to the way humans analyze images. 
How does the application work? The user has the option to either upload an image or select one from the dataset. Our 
model then tries to create a label that is as accurate as possible. To evaluate, the user can vote via a button which 
of the two captions they think better represents the image content. The results are displayed in a graph for viewing 
and further analysis.""")
st.write("")

# Show picture and generate caption
def gen_caption(picture):
    st.image(picture)
    st.subheader('Generated caption:')
    with st.spinner(text='This may take a moment...'):
        caption = predict_step([picture])
    st.write(caption[0])

# User chooses between preuploaded picture or uploads one himself
col1, col2, col3 = st.columns([0.5, 1, 0.5])
with col1:
    pass
with col2:
    st.header('Caption Generator')
    user_choice = st.radio(label='Choose from either option',
                           options=['Upload your own picture', 'Image from our dataset'])
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
with col3:
    pass

if user_choice == 'Upload your own picture':
    picture = st.file_uploader('', type=['jpg'])
    if picture != None:
        with col1:
            pass
        with col2:
            gen_caption(picture)
        with col3:
            pass
        
else:
    # Center the button
    with col1:
        pass
    with col2:
        # Get a random picture from out dataset
        if st.button("Get a picture from our dataset:"):
            picture_list = list(features.keys())
            r = random.randint(0,len(picture_list) - 1)
            pic = picture_list[r]
            picture = images + pic
            gen_caption(picture)
    with col3: 
        pass
