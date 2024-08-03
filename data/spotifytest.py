import cv2
import numpy as np
from keras.models import load_model # type: ignore
import time
from collections import Counter
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Spotify credentials
SPOTIPY_CLIENT_ID = 'd6e7ea6993a04a23a472a597d4fcfd9b'
SPOTIPY_CLIENT_SECRET = 'bcb7251fd0e942db9b1f1368ee17f488'
SPOTIPY_REDIRECT_URI = 'http://localhost:8888/callback/'

# Emotion labels and corresponding Spotify playlists
emotion_playlists = {
    'Angry': 'spotify:playlist:37i9dQZF1DWYzMfRQj22Nd',
    'Disgust': 'spotify:playlist:37i9dQZF1DX3rxVfibe1L0',
    'Fear': 'spotify:playlist:37i9dQZF1DX4WYpdgoIcn6',
    'Happy': 'spotify:playlist:37i9dQZF1DXdPec7aLTmlC',
    'Neutral': 'spotify:playlist:37i9dQZF1DWZeKCadgRdKQ',
    'Sad': 'spotify:playlist:37i9dQZF1DX3YSRoSdA634',
    'Surprise': 'spotify:playlist:37i9dQZF1DX9XIFQuFvzM4'
}

# Initialize Spotify API
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=SPOTIPY_CLIENT_ID,
                                               client_secret=SPOTIPY_CLIENT_SECRET,
                                               redirect_uri=SPOTIPY_REDIRECT_URI,
                                               scope='user-modify-playback-state,user-read-playback-state'))

# Load the trained emotion detection model
model = load_model('model_file_30epochs.h5')

# Load the face detection classifier
faceDetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Emotion labels
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Capture and detect emotion
video = cv2.VideoCapture(0)
emotion_window = []
start_time = time.time()

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 3)

    current_emotion = None

    for (x, y, w, h) in faces:
        sub_face_img = gray[y:y+h, x+x+w]
        resized = cv2.resize(sub_face_img, (48, 48))
        normalize = resized / 255.0
        reshaped = np.reshape(normalize, (1, 48, 48, 1))
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]
        current_emotion = labels_dict[label]

        # Append current emotion to the window
        emotion_window.append(current_emotion)

        # Draw rectangles and text on the image
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, current_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if time.time() - start_time > 5:  # Analyze emotions for 5 seconds
        if emotion_window:
            most_common_emotion = Counter(emotion_window).most_common(1)[0][0]
            playlist_uri = emotion_playlists.get(most_common_emotion)
            if playlist_uri:
                print(f"Playing playlist: {playlist_uri}")  # Print playlist URI for debugging
                # Play the detected emotion playlist
                sp.start_playback(context_uri=playlist_uri)
            break

    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
