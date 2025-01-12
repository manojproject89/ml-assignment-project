from tensorflow import keras



def pre_process_data(data_path, input_shape: tuple, batch_size: int):
    train_dataset = keras.preprocessing.image_dataset_from_directory(
        data_path,
        validation_split=0.2,
        subset="training",
        seed=1337,
        image_size=input_shape[:2],
        batch_size=batch_size,
    )

    val_dataset = keras.preprocessing.image_dataset_from_directory(
        data_path,
        validation_split=0.2,
        subset="validation",
        seed=420,
        image_size=input_shape[:2],
        batch_size=batch_size,
    )

    data_augmentation = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.1),
        ]
    )

    augmented_train_dataset = train_dataset.map(
        lambda x, y: (data_augmentation(x, training=True), y))
    return augmented_train_dataset, val_dataset