import tifffile as tiff
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lr_train import y_test
from lr_train import mean_linear_predictions

# Load the pixel values from the TIF files
true_color_rgb_image = tiff.imread('true_color_RGB_image.tif')

# Reshape the image to match the pixel pairs
true_color_rgb_image = true_color_rgb_image.reshape(-1, 3)

# Evaluate model performance
linear_mse = mean_squared_error(y_test, mean_linear_predictions)
linear_mae = mean_absolute_error(y_test, mean_linear_predictions)
linear_r2 = r2_score(y_test, mean_linear_predictions)

print("Linear Regression MSE:", linear_mse)
print("Linear Regression MAE:", linear_mae)
print("Linear Regression R2 Score:", linear_r2)
print("--------------------------------------------------")


