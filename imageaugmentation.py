import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# Define paths and parameters
original_dataset_dir = 'model_pics_org\sad'
augmented_dataset_dir = 'model_pics_aug\sad'
num_augmented_images_per_original = 5  # Adjust as needed

# Create directories for augmented images if they don't exist
if not os.path.exists(augmented_dataset_dir):
    os.makedirs(augmented_dataset_dir)

# Define augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Iterate over each image in the original dataset
for root, dirs, files in os.walk(original_dataset_dir):
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):  # Adjust based on your image format
            image_path = os.path.join(root, file)
            img = load_img(image_path)  # Load image
            x = img_to_array(img)  # Convert to numpy array (shape: (height, width, channels))
            x = x.reshape((1,) + x.shape)  # Reshape to (1, height, width, channels) for flow()

            # Generate augmented images
            i = 0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=augmented_dataset_dir, save_prefix='aug', save_format='jpeg'):
                i += 1
                if i >= num_augmented_images_per_original:
                    break  # Stop augmentation after reaching the desired number of images per original

print("Data augmentation complete.")
