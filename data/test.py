import cv2
import numpy as np
from keras.models import load_model # type: ignore
import requests
import webbrowser
import time

# Load the trained emotion detection model
model = load_model('model_file_30epochs.h5')

# Load the face detection classifier
faceDetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Emotion labels
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Function to get YouTube recommendations based on emotion and gender
def get_youtube_recommendations(emotion, gender, api_key):
    # Define music queries based on emotions
    emotion_queries = {
        'Angry': 'Energetic rock music',
        'Disgust': 'Relaxing music for discomfort',
        'Fear': 'Calming music for anxiety',
        'Happy': 'Upbeat pop music',
        'Neutral': 'Mellow background music',
        'Sad': 'Soothing sad music',
        'Surprise': 'Exciting and dynamic music'
    }
    
    # Select the query based on detected emotion
    query = emotion_queries.get(emotion, 'Relaxing music')
    
    # Adjust the query based on gender
    if gender == 'Male':
        query += ' for men'
    else:
        query += ' for women'
    
    url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={query}&key={api_key}&maxResults=1"
    response = requests.get(url)
    
    if response.status_code == 200:
        results = response.json().get('items', [])
        if results:
            video_id = results[0]['id'].get('videoId', None)  # Safely get 'videoId'
            if video_id:
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                return video_url
    return None

# Function to detect gender (dummy implementation, replace with actual gender detection)
def detect_gender(frame):
    # Placeholder function; Implement gender detection logic here
    return 'Female'  # For testing purposes, let's assume 'Female'

# Your YouTube Data API key
api_key = 'AIzaSyAZQ0DEDTfSFmlbaNTKOYexF2iffP9zds8'

video = cv2.VideoCapture(0)
last_emotion = None
last_time = time.time()
youtube_open = False

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 3)
    
    current_emotion = None
    
    for x, y, w, h in faces:
        sub_face_img = gray[y:y+h, x:x+w]
        resized = cv2.resize(sub_face_img, (48, 48))
        normalize = resized / 255.0
        reshaped = np.reshape(normalize, (1, 48, 48, 1))
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]
        current_emotion = labels_dict[label]
        
        # Draw rectangles and text on the image
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, f"{current_emotion}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    if current_emotion and current_emotion != last_emotion:
        # Update emotion detection timestamp
        last_time = time.time()
        last_emotion = current_emotion
    
    # Check if enough time has passed and if the emotion is stable
    if (time.time() - last_time) > 3 and current_emotion == last_emotion:
        if not youtube_open:
            # Detect gender
            gender = detect_gender(frame)  # Implement your gender detection logic here
            
            # Fetch YouTube recommendation and open it in the browser
            video_url = get_youtube_recommendations(current_emotion, gender, api_key)
            if video_url:
                print(f"Opening: {video_url}")  # Print URL for debugging
                webbrowser.open(video_url)
                youtube_open = True
        else:
            # Close camera window if YouTube is open
            video.release()
            cv2.destroyAllWindows()
            break
    else:
        # Reset YouTube status if the emotion changes or the time is not yet sufficient
        youtube_open = False
    
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
