import streamlit as st

st.markdown("<h1 style='text-align: center'> Modelling and Evaluation</h1>", unsafe_allow_html=True)

st.subheader('First model')
with st.expander('Word embedding'):
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

with st.expander('Modelling'):
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
with st.expander('Training'):
    st.write("After the finished construction of your model, we compiled it with the categorical cross-entropy "
             "loss function and the Adam optimizer. We then trained the model for 20 epochs, which we felt would be "
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
with st.expander("Generating predictions"):
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
with st.expander("First model evaluation and conclusion"):
    st.write("""The model was capable of producing accurate captions most of the time, however we were not yet satisfied
    with the model. This in hindsight was probably caused by training the model for not long enough. Due to our
    dissatisfaction with the model we evaluated it and came to the conclusion, that the model was simply not accurate 
    enough, which is why we archived it in our repository and went on the search for another approach.""")
    st.subheader("Example caption:")
    st.image("example.jpg")

st.subheader('Second model')
with st.expander("Flax"):
    st.write("""As we were not fully satisfied with our first model and didn't find a more suitable model, whose 
    training would have been feasible (time and hardware wise) for us, we opted to use the pre-trained FlaxVisionEncoderDecoder
     Framework. This is a Transformers library based on JAX/Flax. JAX/Flax allows tracing pure functions and compiling 
     them into efficient, fused accelerator code on both GPU and TPU. Models written in JAX/Flax are immutable and updated
      in a purely functional way which enables simple and efficient model parallelism. """)
with st.expander("Fine tuning Flax"):
    st.write("""The FlaxVisionEncoderDecoder Framework was fine tuned on the MS Coco datset, which like the Flickr 
    dataset contains images with five corresponding captions. The main difference is, that the MS Coco datset needs no 
     Data Cleaning Process, as it is already provided in lower case without punctuations. """)
    st.code("""if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            keep_in_memory=False,
            data_dir=data_args.data_dir,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        dataset = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)""")
    st.write("""Preprocessing the dataset and tokenizing the inputs and targets""")
    st.code("""if training_args.do_train:
        column_names = dataset["train"].column_names
    elif training_args.do_eval:
        column_names = dataset["validation"].column_names
    elif training_args.do_predict:
        column_names = dataset["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    dataset_columns = image_captioning_name_mapping.get(data_args.dataset_name, None)
    if data_args.image_column is None:
        assert dataset_columns is not None
        image_column = dataset_columns[0]
    else:
        image_column = data_args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{data_args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.caption_column is None:
        assert dataset_columns is not None
        caption_column = dataset_columns[1]
    else:
        caption_column = data_args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{data_args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )""")

    st.write("""The pre-trained model was fine tuned via a conventional training loop and an evaluation loop.""")
    st.code("""#the pre-trained Flax model
     model = FlaxVisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder_pretrained_model_name_or_path=model_args.encoder_model_name_or_path,
        decoder_pretrained_model_name_or_path=model_args.decoder_model_name_or_path,
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        encoder_seed=training_args.seed,
        decoder_seed=training_args.seed,
        encoder_dtype=getattr(jnp, model_args.dtype),
        decoder_dtype=getattr(jnp, model_args.dtype),
    )
""")
    st.write("""https://huggingface.co/ydshieh/flax-vision-encoder-decoder-vit-gpt2-coco-en/blob/6b617007a2412a500493cc7ab8737720212286ce/run_image_captioning_flax.py
     this is complete fine tuning process, it contains alot of familiar steps, as well as many new and interesting methods.""")

with st.expander("Implementing FlaxVisionEncoderDecoder Model"):
    st.write("""First a provided picture is transformed into RGB if it is in another image mode such as P. 
                 The feature extractor then transforms the picture into image vectors, which are then passed through the
                 model, which finally goes through the tokenizer to produce a predicted caption.""")
    st.code("""model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
def predict_step(image_paths):
  images = []
  for image_path in image_paths:
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

    images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return preds""")
with st.expander("Evaluating Flax"):
    st.write("""The FlaxVisionEncoderDecoder Model completely outdid our expectations both in accuracy and precision.
     Try out the model yourself on our app page! """)
    
