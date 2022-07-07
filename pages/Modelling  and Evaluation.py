import streamlit as st

st.markdown("<h1 style='text-align: center'> Our Data Preparation Process</h1>", unsafe_allow_html=True)

st.subheader('First model')
with st.expander('word embedding'):
    st.write("""To help our model with generating captions, we utilized the pre-trained GloVe: Global Vector for Word
    Representation. This vector was then used to create an embedding matrix of an uniformly distributed 200 length for 
    all the unique words.""")
    st.code("""embeddings_index = {} # empty dictionary
f = open('glove.6B.200d.txt', encoding="utf-8")

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

embedding_dim = 200

# Get 200-dim dense vector for each of the 10000 words in out vocabulary
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in wordtoix.items():
    #if i < max_words:
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in the embedding index will be all zeros,
        embedding_matrix[i] = embedding_vector
        
embedding_matrix.shape""")

with st.expander('modelling'):
    st.write("""Then we built the LSTM and embedding layers. The model utilizes both the image vectors and word 
    embeddings as inputs. The embedding vector goes through an LSTM architecture, which then makes the appropriate
    word predictions. The image vector is then combined with the LSTM predictions and passed through dense layers and
    a SoftMAx activation function.""")
    st.code("""from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPool2D, LSTM, add
from tensorflow.keras.layers import Activation, Dropout, Flatten, Embedding
from tensorflow.keras.models import Model

inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)

decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.summary()""")
with st.expander('training'):
    st.write("After the finished construction of your model, we compiled it with the categorical cross-entropy "
             "loss function and the Adam optimizer. We then trained the model for 20 epochs, which we felt would be"
             "time appropriate. In hindsight this choice might have been one of the main reasons, why we were "
             "unsatisfied with the trained model in the end. After each the training loop the model was saved via "
             "Keras.")
    st.code("""model.layers[2].set_weights([embedding_matrix])
model.layers[2].trainable = False

model.compile(loss='categorical_crossentropy', optimizer='adam')

epochs = 20
number_pics_per_bath = 3
steps = len(data2)//number_pics_per_bath

features = pickle.load(open("images1.pkl", "rb"))

# https://stackoverflow.com/questions/58352326/running-the-tensorflow-2-0-code-gives-valueerror-tf-function-decorated-functio

tf.config.run_functions_eagerly(True)

for i in range(epochs):
    generator = data_generator(data2, features, wordtoix, max_length, number_pics_per_bath)
    model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    model.save('model_' + str(i) + '.h5')""")
with st.expander("generating predictions"):
    st.write("""To use our model in streamlit we loaded the final saved state after the training loop.""")
    st.code("""features = pickle.load(open("images1.pkl", "rb"))
model = load_model('model_9.h5')
images = "Images/"
max_length = 33
words_to_index = pickle.load(open("words.pkl", "rb"))
index_to_words = pickle.load(open("words1.pkl", "rb"))""")
    st.write("""We then defined a function, that would load the images vector, then create a word embedding and finally
    load both into the trained model, generating a caption.""")
    st.code("""def Image_Caption(picture):
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
    return final""")
    st.write("z representes the index of a picture in our dataset")
    st.code("""pic = list(features.keys())[z]
image = features[pic].reshape((1,2048))
x = plt.imread(images+pic)
plt.imshow(x)
plt.show()
print("Caption:", Image_Caption(image))""")
with st.expander("First model conclusion"):
    st.write("""The model was capable of producing accurate captions most of the time, however we were not yet satisfied
    with the model. This in hindsight was probably caused by training the model for not long enough.""")
    st.subheader("Example caption:")
    st.image("WhatsApp Image 2022-06-20 at 17.39.57.jpeg")
    
