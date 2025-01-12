import os
import pytest
import tensorflow as tf
from tensorflow import keras
from src.preprocess import pre_process_data
from src.model import model_instantiate
from src.train import train_model
from src.inference import model_predict, model_evaluate
from datetime import datetime
# Test Model
# @pytest
@pytest.mark.parametrize(
    "model_name, data_path, input_shape, learning_rate, num_epochs, batch_size",
    [
        # ["./data/cats_dogs/", (224, 224, 3), 0.01,  50, 32],
        # ["custom_net", "./data/cats_dogs/", (224, 224, 3), 0.01,  10, 32],
        # ["mobilenet", "./data/cats_dogs/", (224, 224, 3), 0.01,  10, 32],
        ["mobilenetv2", "./data/cats_dogs/", (224, 224, 3), 0.01,  10, 32],
    ],
)
def test_train_and_evaluate_model(model_name, data_path, input_shape, learning_rate, num_epochs, batch_size):
    experiment_name = "cat-dog-classifier-mobilenet"
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    train_dataset, val_dataset = pre_process_data(data_path=data_path, input_shape=input_shape, batch_size=batch_size)

    model = model_instantiate(model_string=model_name,input_shape=input_shape, learning_rate=learning_rate)

    logdir = os.path.join("logs", experiment_name, run_name)
    tb_callback = keras.callbacks.TensorBoard(log_dir=logdir, write_graph=True, histogram_freq=1)

    history = train_model(model, augmented_train_dataset=train_dataset, num_epochs=num_epochs, val_dataset=val_dataset, tb_callback=tb_callback)
    model_evaluate(val_image_path=val_dataset, model=model)
    test_img_path = os.path.join(data_path, "cat/cat.4060.jpg")
    model_predict(test_image_path=test_img_path, input_shape=input_shape, model=model)
