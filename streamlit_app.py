import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


dataset = pd.read_csv('cardiac_arrest_dataset.csv')

# create our training, and validation data. Randomly choose which data is in which set.
x = dataset[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak','slope' ]]
y = dataset['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# create the model
model = tf.keras.models.Sequential()

# scale the input to fit the model
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

#print(x_train_scaled.shape)
# add layers to the model
model.add(keras.layers.Dense(units=64, activation='relu', input_shape=(x_train_scaled.shape[1],)))
model.add(keras.layers.Dense(8, input_dim= 5, activation='relu'))  # Adjust number of neurons if needed
model.add(keras.layers.Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# train the model
model.fit(x_train_scaled, y_train, epochs=40, batch_size=32, validation_split=0.2, verbose=2)

# evaluate the model.
model.evaluate(x_test_scaled, y_test, verbose=2)


st.title("Coronary Artery Disease Predictor")
st.write(
    "Enter your most recent cardiology lab results to get your results now!"
)


