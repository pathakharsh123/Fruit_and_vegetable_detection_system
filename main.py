import streamlit as st
import tensorflow as tf
import numpy as np
# Tensorflow model prediction


def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.h5")
    image = tf.keras.preprocessing.image.load_img(
        test_image, target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # converting single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)


# sidebar
st.sidebar.title("Dashboard")

app_mode = st.sidebar.selectbox(
    "select page", ["Home", "About Project", "Prediction"])

# Main Page

if (app_mode == "Home"):
    st.header("FRUITS & VEGETABLES DETECTION SYSTEM")
    img_path = "home.jpg"
    st.image(img_path)
elif (app_mode == "About Project"):
    st.header("About Project")

    st.subheader("Overview")
    st.markdown("""
    The Fruits & Vegetables Detection System is a machine learning application designed to accurately identify various types of fruits and vegetables from images.This system offers a reliable solution for automating produce recognition tasks.
    """)

    st.subheader("Dataset")
    st.markdown("""
    **Dataset Source**: [Fruit and Vegetable Image Recognition Dataset](https://www.kaggle.com/username/fruits-vegetables-image-recognition)

    The dataset contains images of the following food items:
    
    - **Fruits**: Banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango.
    - **Vegetables**: Cucumber, carrot, capsicum, onion, potato, lemon, tomato, radish, beetroot, cabbage, lettuce, spinach, soybean, cauliflower, bell pepper, chili pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalapeño, ginger, garlic, peas, eggplant.
    """)

    st.subheader("Dataset Details")
    st.markdown("""
    The dataset consists of three main folders:
    
    1. **Train**: Contains 100 images of each class for training the model.
    2. **Validation**: Contains 10 images of each class for validation during model training.
    3. **Test**: Contains 10 images of each class for evaluating the trained model's performance.
    """)

    st.subheader("Model and Application")
    st.markdown("""
    The system utilizes a convolutional neural network (CNN) architecture trained on TensorFlow and Keras. 
    Additionally, a web application is developed using Streamlit, providing users with a simple interface to interact with the model and get predictions on uploaded images.
    """)
# Prediction
elif (app_mode == "Prediction"):
    st.header("Model Prediction")
    test_image = st.file_uploader("Choose any Image:")
    if (st.button("Show Image")):
        st.image(test_image, width='4', use_column_width=True)
    # Predict button
    if (st.button("Predict")):
        st.balloons()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        with open('labels.txt') as f:
            content = f.readlines()
        labels = []
        for i in content:
            labels.append(i[:-1])
        st.success("Model is Predicting it's a {} .".format(
            labels[result_index]))
