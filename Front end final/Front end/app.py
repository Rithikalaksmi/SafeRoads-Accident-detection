import sys
import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for

# Add the paths for detection and prediction modules
sys.path.append("C:/Users/rithi/OneDrive/Documents/sem 5/machine learning lab/project")
sys.path.append("C:/Users/rithi/OneDrive/Documents/sem 5/machine learning lab/project/Accident-Detection-System-main")

# Importing necessary modules
from detection import AccidentDetectionModel  
import pred  # For prediction functionality

# Initialize the Flask app
app = Flask(__name__)

# Initialize the accident detection model
model = AccidentDetectionModel(
    "C:/Users/rithi/OneDrive/Documents/sem 5/machine learning lab/project/Accident-Detection-System-main/model.json", 
    "C:/Users/rithi/OneDrive/Documents/sem 5/machine learning lab/project/Accident-Detection-System-main/model_weights.weights.h5"
)
font = cv2.FONT_HERSHEY_SIMPLEX

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Statistics page
@app.route('/statistics')
def statistics():
    return render_template('statistics.html')

# Prediction page
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        location = request.form['location']
        date = request.form['date']
        time = request.form['time']

        # Call the prediction function from pred.py
        result = pred.make_prediction(location, date, time)

        # Redirect to the results page with prediction results
        return render_template('result.html', result=result)

    return render_template('prediction.html', result=None)

# Live feed page for video upload and accident detection
@app.route('/live_feed', methods=['GET', 'POST'])
def live_feed():
    if request.method == 'POST':
        # Check if a video file has been uploaded
        if 'file' not in request.files:
            return "No file part in request"

        file = request.files['file']
        if file.filename == '':
            return "No selected file"

        # Save the uploaded video file
        upload_folder = 'uploads'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        video_path = os.path.join(upload_folder, file.filename)
        file.save(video_path)

        # Process the uploaded video for accident detection
        process_video(video_path)

        return redirect(url_for('live_feed'))

    return render_template('live_feed.html')

# Function to process video and detect accidents
def process_video(video_path):
    video = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not video.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = video.read()

        if not ret:
            print("Error: Failed to read frame (end of video or error).")
            break

        try:
            # Convert the frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize the frame for model input
            roi = cv2.resize(rgb_frame, (250, 250))

            # Predict accident using the model
            pred, prob = model.predict_accident(roi[np.newaxis, :, :])

            # Display prediction and probability if an accident is detected
            if pred == "Accident":
                prob = round(prob[0][0] * 100, 2)

                if prob > 90:
                    os.system("say beep")  # Beep alert if the probability is high

                # Draw rectangle and put text on the frame
                cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
                cv2.putText(frame, f"{pred} {prob}%", (20, 30), font, 1, (255, 255, 0), 2)

            # Display the video frame
            cv2.imshow('Video', frame)

            # Exit when 'q' is pressed
            if cv2.waitKey(33) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"Error during prediction: {e}")
            break

    video.release()
    cv2.destroyAllWindows()

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
