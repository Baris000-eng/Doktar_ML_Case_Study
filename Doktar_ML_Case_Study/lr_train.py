import numpy as np
import tifffile as tiff
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the pixel values from the TIF files
raw_rgb_image = tiff.imread('raw_RGB_image.tif')
true_color_rgb_image = tiff.imread('true_color_RGB_image.tif')

# Reshape the images to match the pixel pairs
raw_rgb_image = raw_rgb_image.reshape(-1, 3)
true_color_rgb_image = true_color_rgb_image.reshape(-1, 3)


# Define the mapping function from int16 to int8
def map_int16_to_int8(int16_values):
    return (int16_values / 7000 * 255).astype(np.uint8)


# Map the int16 values in raw_rgb_image to int8
mapped_raw_rgb_image = map_int16_to_int8(raw_rgb_image)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(mapped_raw_rgb_image, true_color_rgb_image, test_size=0.25,
                                                    random_state=43)

# Define the number of bootstrap samples
n_bootstrap_samples = 3

# Lists to store predictions from each bootstrap sample
linear_predictions = []

for _ in range(n_bootstrap_samples):
    # Create a bootstrap sample
    indices = np.random.choice(range(len(X_train)), size=len(X_train), replace=True)
    X_bootstrap = X_train[indices]
    y_bootstrap = y_train[indices]

    # Linear Regression Model
    linear_model = LinearRegression()
    linear_model.fit(X_bootstrap, y_bootstrap)
    linear_pred = linear_model.predict(X_test)
    linear_predictions.append(linear_pred)

# Convert prediction lists to arrays
linear_predictions = np.array(linear_predictions)

# Compute mean prediction across bootstrap samples
mean_linear_predictions = np.mean(linear_predictions, axis=0)

# Round the mean predictions to the nearest integers
mean_linear_predictions = np.round(mean_linear_predictions).astype(np.uint8)
