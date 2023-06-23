import numpy as np
import tifffile as tiff
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from nn_train import mean_nn_predictions
from nn_train import y_test

# Load the pixel values from the TIF files
true_color_rgb_image = tiff.imread('true_color_RGB_image.tif')

# Reshape the image to match the pixel pairs
true_color_rgb_image = true_color_rgb_image.reshape(-1, 3)

# Evaluate model performance
nn_mse = mean_squared_error(y_test, mean_nn_predictions)
nn_mae = mean_absolute_error(y_test, mean_nn_predictions)
nn_r2 = r2_score(y_test, mean_nn_predictions)

print("Neural Network MSE:", nn_mse)
print("Neural Network MAE:", nn_mae)
print("Neural Network R2 Score:", nn_r2)

