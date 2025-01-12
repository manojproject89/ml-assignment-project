# import joblib
# import numpy as np


# # Load the model and scaler
# def load_model(model_path):
#     """Load the trained model from a file."""
#     try:
#         model = joblib.load(model_path)
#         print("Model loaded successfully.")
#         return model
#     except FileNotFoundError:
#         print(f"Model file not found at {model_path}. Please ensure the model is trained and saved.")
#         exit(1)


# def load_scaler(scaler_path):
#     """Load the scaler from a file."""
#     try:
#         scaler = joblib.load(scaler_path)
#         print("Scaler loaded successfully.")
#         return scaler
#     except FileNotFoundError:
#         print(f"Scaler file not found at {scaler_path}. Please ensure the scaler is saved.")
#         exit(1)


# def preprocess_input(input_features, scaler):
#     """Preprocess the input features to match the model's expected format."""
#     input_array = np.array(input_features).reshape(1, -1)
#     print("Original input array shape:", input_array.shape)
    
#     # Ensure the input data has the correct number of features (31)
#     if input_array.shape[1] != 30:  # Adjust this number to match the training feature count
#         raise ValueError(f"Expected 30 features, but got {input_array.shape[1]} features.")
    
#     # Apply scaling transformation using the loaded scaler
#     return scaler.transform(input_array)


# def predict(model, input_features, scaler):
#     """Make predictions using the trained model."""
#     print("Input features:", input_features)
#     input_array = preprocess_input(input_features, scaler)  # Preprocess input before prediction
#     prediction = model.predict(input_array)
#     print("Prediction:", prediction)
#     return prediction


# def main():
#     model_path = "artifacts/model.joblib"
#     scaler_path = "artifacts/scaler.joblib"

#     # Load the model and scaler
#     model = load_model(model_path)
#     scaler = load_scaler(scaler_path)

#     # Define the input features (ensure these match the expected format)
#     # 0
#     example_input_1 = [19.27,26.47,127.9,1162.0,0.09401,0.1719,0.1657,0.07593,0.1853,0.06261,0.5558,0.6062,3.528,68.17,0.005015,0.03318,0.03497,0.009643,0.01543,0.003896,24.15,30.9,161.4,1813.0,0.1509,0.659,0.6091,0.1785,0.3672,0.1123]
#     # 1
#     example_input_2 = [8.196,16.84,51.71,201.9,0.086,0.05943,0.01588,0.005917,0.1769,0.06503,0.1563,0.9567,1.094,8.205,0.008968,0.01646,0.01588,0.005917,0.02574,0.002582,8.964,21.96,57.26,242.2,0.1297,0.1357,0.0688,0.02564,0.3105,0.07409]
#     # 0
#     example_input_3 = [18.25,19.98,119.6,1040.0,0.09463,0.109,0.1127,0.074,0.1794,0.05742,0.4467,0.7732,3.18,53.91,0.004314,0.01382,0.02254,0.01039,0.01369,0.002179,22.88,27.66,153.2,1606.0,0.1442,0.2576,0.3784,0.1932,0.3063,0.08368]
#     # 1
#     example_input_4 = [13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259]
#     # Choose which input to use
#     example_input = example_input_4  # Change this to example_input_2 for the second example
#     inputs = [example_input_1, example_input_2, example_input_3, example_input_4]
#     # Make prediction
#     for i in range(4):
#         prediction = predict(model, inputs[i], scaler)
#         print(f"Predicted class: {prediction[0]}")


# if __name__ == "__main__":
#     main()
from tensorflow import keras
import tensorflow as tf

def model_evaluate(val_image_path, model):
    test_loss, test_accuracy = model.evaluate(val_image_path, steps=50)
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

def model_predict(test_image_path, input_shape, model):
    img = keras.preprocessing.image.load_img(
        test_image_path, target_size=input_shape
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    print("This image is {:.2f}% cat and {:.2f}% dog.".format(100 * float(predictions[0][0]),
                                                            100 * float(predictions[0][1])))

