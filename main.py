import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.layers import GlobalMaxPooling2D

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

# Constructing the model using tf.keras.Sequential
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filename = pickle.load(open('filenames.pkl','rb'))

st.title('Fashion Recommender System')

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0


def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocess_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocess_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices


#file upload -> save
uplodaed_file = st.file_uploader("Choose an Image")
if uplodaed_file is not None:
    if save_uploaded_file(uplodaed_file):
        #display the file
        display_image = Image.open(uplodaed_file)
        st.image(display_image)
        # load file -> feature extract
        features = feature_extraction(os.path.join("uploads", uplodaed_file.name),model)
        # st.text(features)
        # recommendation
        indices = recommend(features,feature_list)
        # show
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.image(filename[indices[0][0]])
        with col2:
            st.image(filename[indices[0][1]])
        with col3:
            st.image(filename[indices[0][2]])
        with col4:
            st.image(filename[indices[0][3]])
        with col5:
            st.image(filename[indices[0][4]])
    else:
        st.header("Some error occured in file uploaded")

