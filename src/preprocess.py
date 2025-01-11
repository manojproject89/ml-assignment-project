import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    """Load the wine quality dataset."""
    data = pd.read_csv(file_path, sep=',')
    return data


def preprocess_data(data):
    """Preprocess the data by splitting into features  target and scaling."""
    X = data.drop([data.columns[0], data.columns[-1]], axis=1)
    y = data['target']
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)

    # Standard scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler
