import streamlit as st

st.markdown("<h1 style='text-align: center'> Our Data Preparation Process</h1>", unsafe_allow_html=True)

with st.expander("Information in advance"):
    st.write("""At the beginning of the project, each of us had chosen a different approach or model, so we could later decide on the best one.
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
    st.write("""As explained in the previous step, we used the Fickr 8k dataset. This consists of 8,000 images, each one
     of the images is paired with five different captions. This is advantageous for our model, as it helps in creating a
      more accurate word embedding. """)

with st.expander("Data Cleaning"):
    st.write("""The first step was to create a dictionary for storing the image files and their respective
     captions. Due to five captions per image, we used the images as the key for the captions.""")
    st.code("""def load_captions(info):
    dict_1 = dict()
    count = 0
    for line in info.split('\n'):
        
        splitter = line.split('.jpg,')
#         print(splitter)
        # The image_code and image_captions are the list of the images with their respective captions
        image_code, image_caption = splitter[0], splitter[1]
        
        # Create dictionary
        if image_code not in dict_1:
            dict_1[image_code] = list()
            
        dict_1[image_code].append(image_caption)
        
    return dict_1

data = load_captions(info)""")
    st.write("""Then we converted every letter into lower case, so as to not lead to different word embedding vectors
        for the same word. Words with a length below two are also removed, so no punctuations remain.""")
    st.code("""def cleanse_data(data):
    dict_2 = dict()
    for key, value in data.items():
        for i in range(len(value)):
            lines = ""
            line1 = value[i]
            for j in line1.split():
                if len(j) < 2:
                    continue
                j = j.lower()
                lines += j + " "
            if key not in dict_2:
                dict_2[key] = list()
            
            dict_2[key].append(lines)
            
    return dict_2

data2 = cleanse_data(data)""")
    st.write("""We then updated the captions file.""")
    st.code("""def vocabulary(data2):
    all_desc = set()
    for key in data2.keys():
        [all_desc.update(d.split()) for d in data2[key]]
    return all_desc
    vocabulary_data = vocabulary(data2)
print(len(vocabulary_data))


def save_dict(data2, filename):
    lines = list()
    for key, value in data2.items():
        for desc in value:
            lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

save_dict(data2, 'captions1.txt')
""")

with st.expander("Feature Engineering"):
    st.write(
        """For our Feature Engineering we utilized the Inception V3 model. In order to use this model, we first had 
         to resize our images to a size of 299x299 pixels.""")
    st.code("""def preprocess(image_path):
    # Convert all the images to size 299x299 as expected by the inception v3 model
    img = keras.preprocessing.image.load_img(image_path, target_size=(299, 299))
    # Convert PIL image to numpy array of 3-dimensions
    x = keras.preprocessing.image.img_to_array(img)
    # Add one more dimension
    x = np.expand_dims(x, axis=0)
    # pre-process the images using preprocess_input() from inception module
    x = keras.applications.inception_v3.preprocess_input(x)
    return x""")
    st.code("""
input1 = InceptionV3(weights='imagenet')

model = Model(input1.input, input1.layers[-2].output)
model.summary()""")
    st.write("Finally we saved the processed images as a pickle file for easier loading later on.")
    st.code("""def encode(image):
    image = preprocess(image) # preprocess the image
    fea_vec = model.predict(image) # Get the encoding vector for the image
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) # reshape from (1, 2048) to (2048, )
    return fea_vec
    
 encoding = {}

for i in tqdm(img):
    encoding[i[len(images):]] = encode(i)
    
import pickle

with open("images1.pkl", "wb") as encoded_pickle:
    pickle.dump(encoding, encoded_pickle)""")


with st.expander("Data Splitting"):
    st.write("""For our model we utilized all 8,000 images for the training loop.""")
    
