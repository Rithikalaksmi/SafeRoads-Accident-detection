import cv2
from detection import AccidentDetectionModel
import numpy as np
import os

# Initialize the model
model = AccidentDetectionModel("C:/Users/rithi/OneDrive/Documents/sem 5/machine learning lab/project/Accident-Detection-System-main/model.json", 
                               'C:/Users/rithi/OneDrive/Documents/sem 5/machine learning lab/project/Accident-Detection-System-main/model_weights.weights.h5')
font = cv2.FONT_HERSHEY_SIMPLEX

def startapplication():
    # Load the video (or webcam)
    video = cv2.VideoCapture('C:/Users/rithi/OneDrive/Documents/sem 5/machine learning lab/project/Accident-Detection-System-main/demo1.mp4')  # For camera, use video = cv2.VideoCapture(0)

    # Check if video opened successfully
    if not video.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = video.read()

        # Check if the frame was successfully read
        if not ret:
            print("Error: Failed to read frame (end of video or error).")
            break

        try:
            # Convert the frame to RGB
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize the frame for model input
            roi = cv2.resize(gray_frame, (250, 250))

            # Predict accident using the model
            pred, prob = model.predict_accident(roi[np.newaxis, :, :])

            # Display prediction and probability if an accident is detected
            if pred == "Accident":
                prob = (round(prob[0][0] * 100, 2))

                # Beep when alert (if prob > 90)
                if prob > 90:
                    os.system("say beep")  # Uncomment to enable beep alert

                # Draw rectangle and put text on the frame
                cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
                cv2.putText(frame, pred + " " + str(prob), (20, 30), font, 1, (255, 255, 0), 2)

            # Display the video frame
            cv2.imshow('Video', frame)

            # Exit when 'q' is pressed
            if cv2.waitKey(33) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"Error during prediction: {e}")
            break

    # Release video capture and close windows
    video.release()
    cv2.destroyAllWindows()

# Corrected main function check
if __name__ == '__main__':
    startapplication()
