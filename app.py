import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Streamlit UI
st.title("Iris Flower Classification with Random Forest")
st.write("This app uses the Iris dataset and a Random Forest classifier.")

st.write(f"### Model Accuracy: {acc:.2f}")

# User input
st.sidebar.header("Input Flower Measurements")
sepal_length = st.sidebar.slider("Sepal Length", float(X[:,0].min()), float(X[:,0].max()))
sepal_width = st.sidebar.slider("Sepal Width", float(X[:,1].min()), float(X[:,1].max()))
petal_length = st.sidebar.slider("Petal Length", float(X[:,2].min()), float(X[:,2].max()))
petal_width = st.sidebar.slider("Petal Width", float(X[:,3].min()), float(X[:,3].max()))

# Prediction
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = clf.predict(input_data)
predicted_class = iris.target_names[prediction[0]]

st.write(f"### Predicted Iris Class: **{predicted_class}**")
