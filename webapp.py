from io import BytesIO
import requests
from PIL import Image
import streamlit as st
import numpy as np
import tensorflow as tf
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Cassava Image Classifier")
st.text("Provide URL of Cassava Image for image classification")


@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('/app/saved_model/')
    return model


with st.spinner('Model is being loaded..'):
    model = load_model()

classes = ['Cassava Bacterial Blight (CBB)', 'Cassava Brown Streak Disease (CBSD)',
           'Cassava Green Mottle (CGM)', 'Cassava Mosaic Disease (CMD)', 'Healthy', 'UNKNOWN']


def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    return np.expand_dims(img, axis=0)


path = st.text_input('Enter image URL to classify..',
                     "https://img.freepik.com/premium-photo/raindrops-surface-cassava-leaves-morning-dew-rain-nature-concept-background_717054-137.jpg?w=900")

if path is not None:
    content = requests.get(path).content

    st.write("Predicted Class:")
    with st.spinner('Wait for it...'):
        label = np.argmax(model.predict(decode_img(content)), axis=1)
        st.write(classes[label[0]])

    st.write("")
    image = Image.open(BytesIO(content))
    st.image(image, caption='Classifying Cassava Image..', use_column_width=True)
    
