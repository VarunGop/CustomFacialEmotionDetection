import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Load pre-trained model for emotion recognition
model = load_model('varun_model.h5')

# Define emotions (modify as needed)
emotions_all = ["Angry", "Disgust", "Happy", "Sad"]

# Initialize OpenCV's Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Variables to track detected emotions and their confidence values since the last click
detected_emotions = {emotion: [] for emotion in emotions_all}
last_click_position = (-1, -1)

# Mouse click callback function
def on_mouse_click(event, x, y, flags, param):
    global detected_emotions, last_click_position

    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button down event
        # Calculate average confidence value of each detected emotion since the last click
        for emotion in emotions_all:
            if detected_emotions[emotion]:  # Check if there are detected emotions for this category
                confidence_values = [confidence for _, confidence in detected_emotions[emotion]]
                average_confidence = np.mean(confidence_values)
                print(f"Average confidence for {emotion}: {average_confidence:.2f}")

        # Clear detected emotions dictionary and update last click position
        detected_emotions = {emotion: [] for emotion in emotions_all}
        last_click_position = (x, y)

# Capture video from the computer's camera
cap = cv2.VideoCapture(0)  # Use camera index 0 (default webcam)

# Create a window and set the mouse callback function
cv2.namedWindow('Emotion Detection')
cv2.setMouseCallback('Emotion Detection', on_mouse_click)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(45, 45))

    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the face region of interest (ROI)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        # Preprocess the ROI for emotion recognition
        roi = roi_gray.astype('float') / 255.0
        roi = np.expand_dims(roi, axis=0)  # Add batch dimension
        roi = np.expand_dims(roi, axis=-1)  # Add channel dimension

        # Predict emotion
        preds = model.predict(roi)[0]
        emotion_label = emotions_all[np.argmax(preds)]
        confidence = preds[np.argmax(preds)]

        # Store detected emotions and their confidence values since the last click
        if last_click_position[0] != -1:  # Check if there was a previous click
            detected_emotions[emotion_label].append((emotion_label, confidence))

        # Draw a rectangle around the face and display the predicted emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"{emotion_label} ({confidence:.2f})", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # Draw the emotion table on the top left corner of the frame
    table_y = 30
    for emotion in emotions_all:
        if detected_emotions[emotion]:  # Check if there are detected emotions for this category
            confidence_values = [confidence for _, confidence in detected_emotions[emotion]]
            average_confidence = np.mean(confidence_values)
            table_str = f"{emotion}: {average_confidence:.2f}"
        else:
            table_str = f"{emotion}: -"
        cv2.putText(frame, table_str, (10, table_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        table_y += 30

    # Display the frame with face detection and predicted emotions
    cv2.imshow('Emotion Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
