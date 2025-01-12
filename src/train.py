import os
import joblib
# from preprocess import load_data, preprocess_data
# from model import train_model, evaluate_model


# def main():
#     if not os.path.exists('artifacts'):
#         os.makedirs('artifacts')
#     # Load and preprocess data
#     data = load_data("data/breast_cancer_data.csv")
#     X_train, X_test, y_train, y_test, scaler = preprocess_data(data)
#     print("Training feature shape:", X_train.shape)
#     # Train the model
#     model = train_model(X_train, y_train)

#     # Evaluate the model
#     accuracy, report = evaluate_model(model, X_test, y_test)
#     print(f"Accuracy: {accuracy}")
#     print(f"Classification Report:\n{report}")
#     joblib.dump(model, "artifacts/model.joblib")
#     print("Model saved to artifacts/model.joblib")
#     joblib.dump(scaler, "artifacts/scaler.joblib")


# if __name__ == "__main__":
#     main()

# Train the model and log metrics and the model itself to MLflow
def train_model(model, augmented_train_dataset, num_epochs, val_dataset, tb_callback):
    if not os.path.exists('artifacts'):
        os.makedirs('artifacts')
    history = model.fit(
        augmented_train_dataset,
        epochs=num_epochs,
        validation_data=val_dataset,
        verbose=2,
        callbacks=[tb_callback]
    )
    joblib.dump(model, "artifacts/model.joblib")
    print("Model saved to artifacts/model.joblib")
    return history