import streamlit as st
from sklearn.model_selection import train_test_split
from tensorflow import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# create the front-end to collect user data
st.title("Coronary Artery Disease Predictor")
st.write(
    "Enter your most recent cardiology lab results to get your results now!"
)
with st.sidebar:
    st.header("Please enter your lab data")
    age = st.number_input("Please enter your age")
    sex = st.selectbox('Gender', ('Male', 'Female'))
    chest_pain = st.selectbox('Chest pain type',
                              ('None', 'Typical angina', 'Atypical angina', 'Non-anginal pain', 'Asymptomatic'))
    blood_sugar = st.selectbox('Is your blood sugar > 120 mg/dl?', ('Yes', 'No'))
    restecg = st.selectbox('Resting ECG results', (
        'Normal', 'ST-T wave abnormality (e.g., T wave inversion, ST elevation/depression > 0.05 mV)',
        "Left ventricular hypertrophy (by Estes' criteria"))
    exang = st.selectbox('Exercise-related angina', ('Yes', 'No'))
    st_slope = st.selectbox('ST Slope', ('Upsloping', 'Flat', 'Downsloping'))
    cholesterol = st.number_input("Cholesterol results", min_value=100, max_value=300)
    restbps = st.number_input("Resting systolic blood pressure results", min_value=80, max_value=250)
    max_heart_rate = st.number_input("Maximum exercise-induced heart rate results", min_value=60, max_value=300)
    oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=50.0)
# Change any categorical data into integers.
if sex == 'Male':
    sex_num = 1
else:
    sex_num = 2

if chest_pain == 'None':
    chest_pain_num = 0
elif chest_pain == 'Typical angina':
    chest_pain_num = 1
elif chest_pain == 'Atypical angina':
    chest_pain_num = 2
elif chest_pain == 'Non-anginal pain':
    chest_pain_num = 3
else:
    chest_pain_num = 4

if blood_sugar == 'Yes':
    blood_sugar_num = 1
else:
    blood_sugar_num = 0

if restecg == 'Normal':
    restecg_num = 0
elif restecg == 'ST-T wave abnormality (e.g., T wave inversion, ST elevation/depression > 0.05 mV)':
    restecg_num = 1
else:
    restecg_num = 2

if exang == 'Yes':
    exang_num = 1
else:
    exang_num = 0

if st_slope == 'Upsloping':
    st_slope_num = 0
elif st_slope == 'Downsloping':
    st_slope_num = 2
elif st_slope == 'Flat':
    st_slope_num = 1


# create a dataframe out of user input for use later in prediction.
input_dict = {'age': age, 'sex': sex_num, 'cp': chest_pain_num, 'trestbps': restbps, 'chol': cholesterol,
              'fbs': blood_sugar_num, 'restecg': restecg_num, 'thalach': max_heart_rate, 'exang': exang_num,
              'oldpeak': oldpeak, 'slope': st_slope_num}

input_df = pd.DataFrame(input_dict, index=[0])

# read the data from .csv into a dataframe.
dataset = pd.read_csv('cardiac_arrest_dataset.csv')
# count the number of patients who have CAD.
true_count = 0
false_count = 0
for x in dataset['target']:
    if x == 1:
        true_count += 1
    else:
        false_count += 1
# count the number of patients who are male and female.
male = 0
female = 0
for i in dataset['sex']:
    if i == 1:
        male += 1
    else:
        female += 1
# plot the distributions of age, sex, and presence of coronary artery disease.
age_distribution = plt.hist(dataset['age'])
cad_distribution = plt.pie([true_count, false_count], ['Has CAD', 'No CAD'])
plt.axis('equal')
sex_distribution = plt.pie([male, female], ['Male', 'Female'])
plt.axis('equal')
# calculate the mean and standard deviation of target in order to reverse the scaling in predictions
mean = dataset['target'].mean()
std = dataset['target'].std()


# Below is the code for the model. It has already been trained and saved as 'heart_disease_predicition_model.h5'


# create our training, and validation data. Randomly choose which data is in which set.

x = dataset[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak','slope' ]]
y = dataset['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# create the model
# model = tf.keras.models.Sequential()

# scale the input to fit the model
scaler = StandardScaler()

# x_train_scaled = scaler.fit_transform(x_train)
# x_test_scaled = scaler.fit_transform(x_test)

# print(x_train_scaled.shape)
# add layers to the model
# model.add(keras.layers.Dense(units=64, activation='relu', input_shape=(x_train_scaled.shape[1],)))
# model.add(keras.layers.Dense(8, input_dim= 5, activation='relu'))  # Adjust number of neurons if needed
# model.add(keras.layers.Dense(1, activation='linear'))

# Compile the model
# model.compile(optimizer='adam', loss='mean_squared_error')

# train the model
# model.fit(x_train_scaled, y_train, epochs=40, batch_size=32, validation_split=0.2, verbose=2)

# evaluate the model.
# model.evaluate(x_test_scaled, y_test, verbose=2)


# load the model

prediction_model = tf.keras.models.load_model('heart_disease_prediction_model.h5')

# manipulate data to fit the model and add input data
dataset.drop(columns=['ca', 'thal', 'target'], inplace=True)
dataset = pd.concat([dataset, input_df], axis=0)
dataset_scaled = scaler.fit_transform(dataset)

# make predictions on the dataset
prediction = prediction_model.predict(dataset_scaled)
x_scaled_test = scaler.fit_transform(x_test)
prediction_test = prediction_model.predict(x_test)



# isolate the user prediction from the general dataframe
user_predict = prediction[-1]

## As the model can predict values slightly lower than 0 or slightly higher than 1, we will scale the extreme ends of prediction to < .05 and >.95. This will alleviate user confusion.
if user_predict[0] < .05:
    user_predict[0] = .05
elif user_predict[0] > .95:
    user_predict[0] = .95




# return the model prediction and a message classifying the user as low or high risk.

dataset['prediction'] = prediction
if st.button("Predict"):
    st.write(f"Your likelihood of Coronary Artery Disease: {user_predict}")
    if user_predict[0] > .60:
        st.write("See your doctor- you have a high risk of developing Coronary Artery Disease.")
    else:
        st.write("You have a low risk of developing Coronary Artery Disease.")


st.pyplot(age_distribution)
st.pyplot(cad_distribution)
st.pyplot(sex_distribution)