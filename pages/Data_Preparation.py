import streamlit as st

st.markdown("<h1 style='text-align: center'> Our Data Preparation Process</h1>", unsafe_allow_html=True)

with st.expander("Information"):
    st.write("""At the beginning of the project, each of us had taken a different approach or model so that we could subsequently decide on the best one.
             Since the current module was already excellently pre-trained, the following process refers to a different model that was just less precise.
             The model can also be found in the Github repository.""")

with st.expander("Data Understanding"):
    st.write("""To train, validate and test the model, we were first provided with the LAION dataset, which is available to us as a CSV file. With 240 TB, 
             this is currently the largest freely accessible image-text dataset in the world. Due to the often inaccurate captions and vague image content,
             we searched for other, more suitable datasets to form the basis of our project. After research, we decided on the Flickr dataset, which stores 
             images and captions separately. The dataset contains high-quality images with varying content, but all can be meaningfully classified.""")

    st.subheader("Three example images from the Flickr8k dataset")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("Images/3726130458_07df79e969.jpg", use_column_width='auto')
    
    with col2:
        st.image("Images/3717309680_e5105afa6d.jpg", use_column_width='auto')

    with col3:
        st.image("Images/3715559023_70c41b31c7.jpg", use_column_width='auto')

with st.expander("Data Selection"):
    st.write(""" As explained in the previous step, we used the Fickr 8k dataset. This consists of 8,000 images, 
             which is the total amount we used for training, validation and testing.""")


with st.expander("Data Cleaning"):
    st.write("""Since the Flickr dataset is already designed for our use case, including image captioning, we had to do little preparation of the data.
             For each image, there are 5 different exact captions that we loaded into the dictionary with the image as the key and the caption as the value.
             In addition, the image captions were written in lowercase. The situation was different with the LAION dataset I had tried before. 
             There we cleaned up the labels to English, NSFW, a maximum length of the labels and matching keywords. """)

with st.expander("Feature Engineering"):
    st.write("""For the feature engineering process we used the Inception V3 transfer learning model. Making use of the pre-trained weights on the image net dataset.""")


with st.expander("Data Splitting"):
    st.write("""Of the total 8,000 images, we used 5,000 (62.5%) for training, 2,000 (25%) for validation, and 1,000 (12.5%) for testing.
             The reason for this was the storage capacities in Streamlit as well as the training effort. """)