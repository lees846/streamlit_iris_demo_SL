import streamlit as st
import pandas as pd
import numpy as np
import pickle

df = pd.DataFrame({'flower': ['iris'], 'price': ['$12']})

st.title("Iris Classifier")
st.write("Please enter the measurements of the flower you'd like to classify")

# Create input for each variable
sl = st.number_input('Sepal Length')
sw = st.number_input('Sepal Width')
pl = st.number_input('Petal Length')
pw = st.number_input('Petal Width')

X = np.array([[sl, sw, pl, pw]])

# Read the model
with open('iris_model.pkl', 'rb') as f:
    flower_model = pickle.load(f)

# Make a prediction
prediction = flower_model.predict(X)
iris_dict = {0: 'Setosa',  1: 'Versicolor', 2: 'Virginica'}

# Share the results
st.write(iris_dict[prediction[0]])