# Fruit and Vegetable Recognition System

This project is a machine learning application built using TensorFlow and Keras for fruit and vegetable recognition. The model is trained on a dataset consisting of 36 classes of fruits and vegetables. The dataset includes separate sets for training, validation, and testing.

## Dataset
The dataset used for training the model can be found on Kaggle: [Fruit and Vegetable Image Recognition Dataset](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition)

## Model Accuracy
The trained model achieved an accuracy of 92% on the test set.

## Web Application
The project includes a web application developed using Streamlit, which allows users to upload images of fruits and vegetables to get predictions from the trained model.

### Deployed Application
The web application is deployed on Streamlit Community Cloud. You can access it [here](https://fruitandvegetabledetectionsystem-kjrsl9gqzafgtrotfmuubn.streamlit.app/).

## Instructions for Local Setup
To run the application locally, follow these steps:

1. Clone this repository to your local machine.
2. Install the required dependencies using the following command: pip install -r requirements.txt
3. Run the Streamlit application with the following command: streamlit run main.py
4. Open your web browser and go to `http://localhost:8501` to access the application.

## Usage
1. Upload an image of a fruit or vegetable using the provided interface.
2. Click on the "Predict" button to get the prediction results.
3. The model will classify the uploaded image into one of the 36 classes of fruits and vegetables.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



