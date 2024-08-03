import cv2
import numpy as np
from keras.models import load_model # type: ignore
import requests
import webbrowser
import time

# Load the trained model
model = load_model('model_file_30epochs.h5')

# Emotion labels
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Function to get YouTube recommendations based on emotion
def get_youtube_recommendations(emotion, api_key):
    query = f"{emotion} music"
    url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={query}&key={api_key}&maxResults=1"
    response = requests.get(url)
    if response.status_code == 200:
        results = response.json().get('items', [])
        if results:
            video_url = f"https://www.youtube.com/watch?v={results[0]['id']['videoId']}"
            return video_url
    return None

# Your YouTube Data API key
api_key = 'YOUR_YOUTUBE_API_KEY'

# Main loop
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float')/255.0
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)
            prediction = model.predict(roi)[0]
            max_index = np.argmax(prediction)
            predicted_emotion = class_labels[max_index]
            
            # Display the emotion label
            label_position = (x, y-10)
            cv2.putText(frame, predicted_emotion, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Fetch YouTube recommendation and open in browser
            video_url = get_youtube_recommendations(predicted_emotion, api_key)
            if video_url:
                print(f"Opening: {video_url}")  # Print URL for debugging
                webbrowser.open(video_url)
                time.sleep(10)  # Wait for 10 seconds before making another recommendation
        else:
            cv2.putText(frame, "No Face Found", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
