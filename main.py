import cv2
import time
import mediapipe as mp 
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

hand_option = HandLandmarkerOptions(
    base_options = BaseOptions(model_asset_path = "hand_landmarker.task"),
    running_mode =VisionRunningMode.IMAGE,
    num_hands = 2
)

face_option = FaceLandmarkerOptions(
    base_options = BaseOptions(model_asset_path = "face_landmarker.task"),
    running_mode = VisionRunningMode.IMAGE,
    num_faces = 1
)


detector_hand = HandLandmarker.create_from_options(hand_option)
detector_face = FaceLandmarker.create_from_options(face_option)

p_time = 0

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 9), (9, 13), (13, 17), (17, 0),
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (9, 10), (10, 11), (11, 12),  # Middle
    (13, 14), (14, 15), (15, 16),# Ring
    (17, 18), (18, 19), (19, 20) # Pinky
]

cap = cv2.VideoCapture(0)

while True:
    success, image = cap.read()
    
    if not success:
        break
    
    c_time = time.time()
    fps = 1 / (c_time-p_time)
    p_time = c_time
    
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(mp.ImageFormat.SRGB, rgb)
    
    result_1 = detector_face.detect(mp_image)
    result_2 = detector_hand.detect(mp_image)
    
    if result_2.hand_landmarks:
        for hand in result_2.hand_landmarks:

            h, w, _ = image.shape
            lm_list = []

            # Convert landmarks to pixel coordinates
            for lm in hand:
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            # -------- Draw landmarks --------
            for x, y in lm_list:
                cv2.circle(image, (x, y), 4, (0, 0, 255), cv2.FILLED)
                
            for start, end in HAND_CONNECTIONS:
                cv2.line(image, lm_list[start], lm_list[end], (0, 255, 0), 2)
    
    if result_1.face_landmarks:
        for face in result_1.face_landmarks:
            h, w, _ = image.shape
            
            lm_list = []
            
            for lm in face:
                lm_list.append((int(lm.x*w), int(lm.y*h)))
                
            for x, y in lm_list:
                cv2.circle(image, (x, y), 1, (255, 255, 0), cv2.FILLED)
    
    cv2.putText(image,
                f'FPS = {int(fps)}',
                (10,30),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (255,255,255),
                2
                )
    
    cv2.imshow("Hand and Face Detection", image)
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()