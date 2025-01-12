from tensorflow import keras
from tensorflow.keras.applications import MobileNet, MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization


def custom_model(input_shape):
    model = Sequential([
    # First Convolutional Block
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Second Convolutional Block
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Third Convolutional Block
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Flatten and Fully Connected Layers
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # Two outputs for one-hot encoded labels
])
    return model

def mobilenet(input_shape):
    base_model = MobileNet(input_shape=input_shape, include_top=False, weights="imagenet")
    base_model.trainable = False
    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(2, activation="softmax")
    ])
    return model

def mobilenetv2(input_shape):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    base_model.trainable = False
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),  # Reduce dimensions for a compact representation
        Dense(128, activation='relu'),  # Fully connected layer
        keras.layers.Dense(2, activation="softmax")
    ])
    return model

# Dictionary to map function names to functions
model_map = {
    "custom_net": custom_model,
    "mobilenet": mobilenet,
    "mobilenetv2": mobilenetv2,
}


# Example usage
def model_selector(model_string, input_shape):
    network = model_map.get(model_string)
    if network:
        return network(input_shape)
    else:
        return "Invalid Model!"

def model_instantiate(model_string, input_shape: tuple, learning_rate: float):

    model = model_selector(model_string, input_shape)

    # Compile the model with a loss function and optimizer
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )

    return model