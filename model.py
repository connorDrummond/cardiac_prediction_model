import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import matplotlib.pyplot as plt


dataset = pd.read_csv('cardiac_arrest_dataset.csv')

# calculate the mean and standard deviation of target in order to reverse the scaling in predictions
mean = dataset['target'].mean()
std = dataset['target'].std()

# Below is the code for the model. It has already been trained and saved as 'heart_disease_predicition_model.h5'


# create our training, and validation data. Randomly choose which data is in which set.

x = dataset[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope']]
y = dataset['target']



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# create the model
model = tf.keras.models.Sequential()

# scale the input to fit the model
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)
x_scaled = scaler.fit_transform(x)
# print(x_train_scaled.shape)
# add layers to the model
model.add(keras.layers.Dense(units=64, activation='relu', input_shape=(x_train_scaled.shape[1],)))
model.add(keras.layers.Dense(8, input_dim=5, activation='relu'))  # Adjust number of neurons if needed
model.add(keras.layers.Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# train the model
model.fit(x_train_scaled, y_train, epochs=40
          , batch_size=32, validation_split=0.2, verbose=2)

# evaluate the model.
model.evaluate(x_test_scaled, y_test, verbose=2)

# save the model to upload to the application
#model.save('heart_disease_prediction_model.h5')

# concatenate our predictions along with target to our test set for analysis

test_predict = model.predict(x_test_scaled)
y_predict = pd.DataFrame(y_test, columns=['target'])
x_tst_df = pd.DataFrame(data = x_test_scaled, columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope'])
x_pred = pd.DataFrame(data = test_predict, columns = ['predictions'])
y_predict = y_predict.reset_index(drop=True)

compare_df = pd.concat([x_tst_df, x_pred], axis=1, ignore_index=True)
compare_df = pd.concat([compare_df, y_predict], axis=1, ignore_index=False)
#print(compare_df)

#dfseries = pd.Series(compare_df.flatten(), name='prediction')

#dataset['prediction'] = dfseries
#dataset['rescaled'] = dataset['prediction']*.5+.5

#dataset.to_csv('output.csv')


