#!/usr/bin/env python
# coding: utf-8

# In[1]:

#####Importing necessary libraries for numpy arrays, tiff image reading, and model training.
import numpy as np
import tifffile as tiff
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the pixel values from the TIF files
raw_rgb_image = tiff.imread('raw_RGB_image.tif') ##############reading the tif file of raw rgb image 
true_color_rgb_image = tiff.imread('true_color_RGB_image.tif') #########reading the tif file of true color rgb image.

# Reshape the images to match the pixel pairs
raw_rgb_image = raw_rgb_image.reshape(-1, 3) #########her row rgb channellarını gösterecek şekilde imageları columnar olacak şekilde reshape etme.
true_color_rgb_image = true_color_rgb_image.reshape(-1, 3) #########her row rgb channellarını gösterecek şekilde imageları columnar olacak şekilde reshape etme.

# Define the mapping function from int16 to int8
def map_int16_to_int8(int16_values):
    return (int16_values / 7000 * 255).astype(np.uint8) ###########dividing with 7000 and multiplying with 255 to make the ranges match.

# Map the int16 values in raw_rgb_image to int8
mapped_raw_rgb_image = map_int16_to_int8(raw_rgb_image) 

# Split the data into training and testing sets (test: %25, train: %75, seed is chosen as 43 to obtain consistent results for different runs)
X_train, X_test, y_train, y_test = train_test_split(mapped_raw_rgb_image, true_color_rgb_image, test_size=0.25, random_state=43)

# Define the number of bootstrap samples
n_bootstrap_samples = 3

# Lists to store predictions from each bootstrap sample
linear_predictions = [] #########list for storing the linear regression predictions
nn_predictions = [] ###########list for storing the neural network predictions

for _ in range(n_bootstrap_samples):
    # Create a bootstrap sample, replacement done for sending the selected index back to the set of input indices.
    indices = np.random.choice(range(len(X_train)), size=len(X_train), replace=True) ####randomly select bootstrap indices with a length of x train from the range of 0 to length of x train data.
    X_bootstrap = X_train[indices] ###########get the x train data corresponding to the bootstrap indices
    y_bootstrap = y_train[indices]  ###########get the y train data corresponding to the bootstrap indices

    # Linear Regression Model
    linear_model = LinearRegression()  # Create an instance of the linear regression model
    linear_model.fit(X_bootstrap, y_bootstrap)  # Fit the model using the training data (X_bootstrap, y_bootstrap)
    linear_pred = linear_model.predict(X_test)  # Perform predictions on the test data (X_test)
l    inear_predictions.append(linear_pred)  # Append the predictions to the linear_predictions list

    # Neural Network Model
    model = Sequential() ##creating an instance of the neural network model
    ########input data with 3 features where features correspond to the RGB channels.
    model.add(Dense(32, activation='relu', input_shape=(3,))) #####adding a fully connected dense layer to the neural network model. Rectified Linear Unit is used as the activation function.
    model.add(Dense(32, activation='relu')) #######adding another dense layer with activation function rectified linear unit and number of neurons 32.
    model.add(Dense(3)) ##############adding another dense layer with a number of neurons as 3.
    model.compile(optimizer='adam', loss='mean_squared_error') #####compiling the neural network model with adam optimizer and mean squared error loss.
    model.fit(X_bootstrap, y_bootstrap, epochs=5, batch_size=32) #########fitting the neural network model to the training data with number of epochs as 5 and batch size as 32.
    nn_pred = model.predict(X_test) ###########predicting the neural network outcome on xtest.
    nn_predictions.append(nn_pred) ########appending the neural network prediction to the list which keeps it.

# Convert prediction lists to arrays
linear_predictions = np.array(linear_predictions) ######converting the linear regression predictions to the numpy array
nn_predictions = np.array(nn_predictions) ######converting the nn regression predictions to the numpy array

# Compute mean prediction across bootstrap samples
mean_linear_predictions = np.mean(linear_predictions, axis=0) ########computing the mean of the linear regression predictions 
mean_nn_predictions = np.mean(nn_predictions, axis=0) ########computing the mean of the nn predictions 

# Round the mean predictions to the nearest integers
mean_linear_predictions = np.round(mean_linear_predictions).astype(np.uint8) ########rounding the mean linear predictions to the np.uint8.
mean_nn_predictions = np.round(mean_nn_predictions).astype(np.uint8) ###########rounding the mean nn predictions to the unsigned int8 format.

# Evaluate model performance
linear_mse = mean_squared_error(y_test, mean_linear_predictions) ###calculating mean squared error with mean linear predictions, and ytest.
linear_mae = mean_absolute_error(y_test, mean_linear_predictions) ###calculating mean absolute error with mean linear predictions, and ytest.
linear_r2 = r2_score(y_test, mean_linear_predictions) #############calculating r^2 score with mean linear predictions and ytest.

nn_mse = mean_squared_error(y_test, mean_nn_predictions) ####calculating mean squared error with mean nn predictions and ytest
nn_mae = mean_absolute_error(y_test, mean_nn_predictions) ####calculating mean absolute error with mean nn predictions and ytest
nn_r2 = r2_score(y_test, mean_nn_predictions) ####r^2 score calculation with mean nn predictions and ytest

print()
print()

#####printing mse, mae, and r^2 values#####################
print("Linear Regression MSE:", linear_mse)
print("Linear Regression MAE:", linear_mae)
print("Linear Regression R2 Score:", linear_r2)
print("--------------------------------------------------")

print("Neural Network MSE:", nn_mse)
print("Neural Network MAE:", nn_mae)
print("Neural Network R2 Score:", nn_r2)
#####printing mse, mae, and r^2 values#####################








# In[ ]:




