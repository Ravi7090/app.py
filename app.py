import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

def clean_data(df):
    # Remove leading/trailing whitespace
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Drop duplicate rows (exact duplicates)
    df = df.drop_duplicates()

    # Impute missing values
    imputer = SimpleImputer(strategy='most_frequent')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    return df_imputed

def detect_invalid_entries(df, label_column='is_valid'):
    if label_column not in df.columns:
        raise ValueError("Label column missing for supervised learning.")

    X = df.drop(label_column, axis=1)
    y = df[label_column]

    # Convert categorical variables
    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)
    score = clf.score(X_test, y_test)

    return score, predictions

def detect_duplicates(df):
    # Convert categorical to numeric
    df_encoded = pd.get_dummies(df.select_dtypes(include=['object', 'category']))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_encoded)

    clustering = DBSCAN(eps=0.5, min_samples=2)
    labels = clustering.fit_predict(X_scaled)

    df['Duplicate_Group'] = labels
    return df[df['Duplicate_Group'] != -1]  # -1 is noise (non-duplicate)

