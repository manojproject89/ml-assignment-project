from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

def train_model(X_train, y_train, param_grid):
    """Train a Random Forest classifier."""
    print("Train model")
    model = RandomForestClassifier(random_state=42)
    # Create GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring="accuracy",
        verbose=1,
        n_jobs=-1
    )

    # Perform grid search
    grid_search.fit(X_train, y_train)

    # Best parameters and score
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)
    best_model = grid_search.best_estimator_
    return best_model


def evaluate_model(model, X_test, y_test):
    """Evaluate the model on the test set."""
    print("Evaluating model")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report
