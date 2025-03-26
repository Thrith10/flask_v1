#%% Reference: https://github.com/googlesamples/mediapipe/tree/main/examples/hand_landmarker/raspberry_pi
# Download hand land mark detector model wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

#%% Parameters
numHands = 2 # Number of hands to be detected
model = 'hand_landmarker.task' # Model for finding the hand landmarks Download using wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
minHandDetectionConfidence = 0.5 # Thresholds for detecting the hand
minHandPresenceConfidence = 0.5
minTrackingConfidence = 0.5
frameWidth = 640
frameHeight = 480

# Visualization parameters
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
POINT_COLOR = (0,255,0)
POINT_RADIUS = 3

#%% Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path=model)
options = vision.HandLandmarkerOptions(
        base_options=base_options,        
        num_hands=numHands,
        min_hand_detection_confidence=minHandDetectionConfidence,
        min_hand_presence_confidence=minHandPresenceConfidence,
        min_tracking_confidence=minTrackingConfidence)
detector = vision.HandLandmarker.create_from_options(options)


#%% Open CV Video Capture and frame analysis
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# The loop will break on pressing the 'q' key
while True:
    try:
        # Capture one frame
        ret, frame = cap.read() 
        
        frame = cv2.flip(frame, 1) # To flip the image to match with camera flip
        
        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        
        # Run hand landmarker using the model.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        detection_result = detector.detect(mp_image)
        
        hand_landmarks_list = detection_result.hand_landmarks
        
        #handedness_list = detection_result.handedness # Could be used to check for which hand
        
        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            
            for landmark in hand_landmarks:
                x= int(landmark.x * frame.shape[1])
                y= int(landmark.y * frame.shape[2])
                cv2.circle(frame,(x,y), POINT_RADIUS, POINT_COLOR,-1)
            cv2.putText(frame,f'Hand{idx+1}',(10,50+idx *30), cv2.FONT_HERSHEY_DUPLEX, FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
        cv2.imshow('Annotated Image', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break
   
    except KeyboardInterrupt:
        break

cap.release()
cv2.destroyAllWindows()
