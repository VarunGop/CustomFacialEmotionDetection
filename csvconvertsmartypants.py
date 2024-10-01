import os
import cv2
import pandas as pd

def detect_face_and_crop(image_path, output_size=(48, 48)):
    # Load the image
    img = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use a face cascade (for example, Haar cascade) to detect faces
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        # No faces detected, return None or raise an exception
        return None
    
    # Find the index of the largest face
    largest_face_index = 0
    largest_area = 0
    
    for i, (x, y, w, h) in enumerate(faces):
        area = w * h
        if area > largest_area:
            largest_area = area
            largest_face_index = i
    
    # Extract the coordinates of the largest face
    (x, y, w, h) = faces[largest_face_index]
    face_roi = img[y:y+h, x:x+w]
    
    # Resize the cropped face ROI to the desired output size
    face_resized = cv2.resize(face_roi, output_size)
    
    # Convert to grayscale
    face_greyscale = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    
    return face_greyscale

def convert_images_to_csv(input_folder, output_csv):
    data = []
    
    # Emotion labels corresponding to the filenames (assuming filename format is label_filename.jpg)
    emotion_mapping = {
        'angry': 0,
        'disgust': 1,
        'happy': 2,
        'sad': 3
    }
    
    # Loop through each image in the folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg"):
            image_path = os.path.join(input_folder, filename)
            
            # Detect face and crop the image
            face_img = detect_face_and_crop(image_path)
            
            if face_img is not None:
                # Extract label from filename (assuming filename format is label_filename.jpg)
                label = filename.split('_')[0]  # Extract the emotion label
                
                if label in emotion_mapping:
                    emotion = emotion_mapping[label]  # Map emotion label to corresponding number
                
                    # Flatten the 2D greyscale image into a 1D array (for CSV)
                    img_flattened = face_img.flatten()
                    pixels_str = ' '.join(map(str, img_flattened.tolist()))  # Convert pixel values to space-separated string
                    
                    # Append label (emotion) and flattened image data (pixels) to the list
                    data.append([emotion, pixels_str])
    
    # Create a DataFrame from the data with appropriate column names
    df = pd.DataFrame(data, columns=['emotion', 'pixels'])
    
    # Save DataFrame to CSV
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    # Input folder containing the images
    input_folder = "model_pics"
    
    # Output CSV file path
    output_csv = "varun_pics.csv"
    
    # Convert images to CSV with face detection and cropping
    convert_images_to_csv(input_folder, output_csv)
