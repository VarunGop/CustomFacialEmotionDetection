import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Load the dataset (assuming 'varun_pics.csv' contains 'pixels' and 'emotion' columns)
data = pd.read_csv('varun_pics.csv')

pixels = data['pixels'].tolist()
labels = data['emotion'].tolist()

# Convert pixels to numpy arrays and normalize
X = np.array([np.fromstring(pixel, dtype=int, sep=' ').reshape(48, 48, 1) for pixel in pixels])
X = X / 255.0  # Normalize pixel values

# Convert labels to categorical format
y = to_categorical(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Convert grayscale images (1 channel) to RGB images (3 channels)
X_train_rgb = np.repeat(X_train, 3, axis=-1)
X_test_rgb = np.repeat(X_test, 3, axis=-1)

# Load pre-trained VGG16 model (excluding top layers)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))

# Freeze the pre-trained layers to prevent them from being updated during training
for layer in base_model.layers:
    layer.trainable = False

# Create a new model on top of the pre-trained base model
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))  # Output layer for 4 classes

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_rgb, y_train, epochs=20, validation_data=(X_test_rgb, y_test))

# Save the trained model to an HDF5 file
model.save("varun_model_pain.h5")

# Evaluate the model
loss, accuracy = model.evaluate(X_test_rgb, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()