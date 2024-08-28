import cv2
import numpy as np
from keras.models import load_model  # type: ignore
import time
from collections import Counter
import webbrowser

# AI-themed movie collections based on emotions
emotion_movies = {
    'Angry': [
        'https://www.imdb.com/title/tt0088247/',  # The Terminator
        'https://www.imdb.com/title/tt0343818/',  # I, Robot
        'https://www.imdb.com/title/tt2395427/',  # Avengers: Age of Ultron
        'https://www.imdb.com/title/tt1535108/',  # Elysium
        'https://www.imdb.com/title/tt0070909/'   # Westworld
    ],
    'Disgust': [
        'https://www.imdb.com/title/tt0470752/',  # Ex Machina
        'https://www.imdb.com/title/tt0212720/',  # A.I. Artificial Intelligence
        'https://www.imdb.com/title/tt1856101/',  # Blade Runner 2049
        'https://www.imdb.com/title/tt1219827/',  # Ghost in the Shell
        'https://www.imdb.com/title/tt0242653/'   # The Matrix Revolutions
    ],
    'Fear': [
        'https://www.imdb.com/title/tt0062622/',  # 2001: A Space Odyssey
        'https://www.imdb.com/title/tt1798709/',  # Her
        'https://www.imdb.com/title/tt0133093/',  # The Matrix
        'https://www.imdb.com/title/tt0181689/',  # Minority Report
        'https://www.imdb.com/title/tt6499752/'   # Upgrade
    ],
    'Happy': [
        'https://www.imdb.com/title/tt0910970/',  # WALL-E
        'https://www.imdb.com/title/tt2245084/',  # Big Hero 6
        'https://www.imdb.com/title/tt0371746/',  # Iron Man
        'https://www.imdb.com/title/tt1823672/',  # Chappie
        'https://www.imdb.com/title/tt0091949/'   # Short Circuit
    ],
    'Neutral': [
        'https://www.imdb.com/title/tt1798709/',  # Her
        'https://www.imdb.com/title/tt0182789/',  # Bicentennial Man
        'https://www.imdb.com/title/tt0212720/',  # A.I. Artificial Intelligence
        'https://www.imdb.com/title/tt2209764/',  # Transcendence
        'https://www.imdb.com/title/tt2084970/'   # The Imitation Game
    ],
    'Sad': [
        'https://www.imdb.com/title/tt0083658/',  # Blade Runner
        'https://www.imdb.com/title/tt0182789/',  # The Bicentennial Man
        'https://www.imdb.com/title/tt0212720/',  # A.I. Artificial Intelligence
        'https://www.imdb.com/title/tt2317225/',  # The Machine
        'https://www.imdb.com/title/tt0113568/'   # Ghost in the Shell
    ],
    'Surprise': [
        'https://www.imdb.com/title/tt0470752/',  # Ex Machina
        'https://www.imdb.com/title/tt0133093/',  # The Matrix
        'https://www.imdb.com/title/tt0139809/',  # The Thirteenth Floor
        'https://www.imdb.com/title/tt1856101/',  # Blade Runner 2049
        'https://www.imdb.com/title/tt2397535/'   # Predestination
    ]
}

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
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)  # Adjusted for better face detection

    current_emotion = None

    for (x, y, w, h) in faces:
        sub_face_img = gray[y:y+h, x:x+w]  # Fixed to correctly crop the face
        resized = cv2.resize(sub_face_img, (48, 48))
        normalize = resized / 255.0
        reshaped = np.reshape(normalize, (1, 48, 48, 1))
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]
        current_emotion = labels_dict[label]

        # Append current emotion to the window
        emotion_window.append(current_emotion)

        # Draw rectangles and text on the image
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, current_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # After 5 seconds, analyze the most common emotion and act on it
    if time.time() - start_time > 5:
        if emotion_window:
            most_common_emotion = Counter(emotion_window).most_common(1)[0][0]
            print(f"Most common emotion: {most_common_emotion}")  # Debug print

            movie_urls = emotion_movies.get(most_common_emotion)
            if movie_urls:
                print(f"Recommended movies: {movie_urls}")  # Print movie URLs for debugging
                # Open the first movie URL in the web browser
                webbrowser.open(movie_urls[0])
            break

    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
