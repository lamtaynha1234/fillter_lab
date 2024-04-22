import mediapipe as mp
import cv2
import math
import numpy as np
import faceBlendCommon as fbc
import csv

VISUALIZE_FACE_POINTS = False
 
filters_config = {
    'pig':
        [{'path': "/Users/kaiser/Documents/GitHub/fillter_lab/filter/pig_nose.png",
          'anno_path': "/Users/kaiser/Documents/GitHub/fillter_lab/filter/pig_nose1.csv",
          'morph': False, 'animated': True, 'has_alpha': True}],
}

def getLandmarks(img):
    mp_face_mesh = mp.solutions.face_mesh
    selected_keypoint_indices = [127, 93, 58, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 288, 323, 356, 70, 63, 105, 66, 55,
                 285, 296, 334, 293, 300, 168, 6, 195, 4, 64, 60, 94, 290, 439, 33, 160, 158, 173, 153, 144, 398, 385,
                 387, 466, 373, 380, 61, 40, 39, 0, 269, 270, 291, 321, 405, 17, 181, 91, 78, 81, 13, 311, 306, 402, 14,
                 178, 162, 54, 67, 10, 297, 284, 389]
 
    height, width = img.shape[:-1]
 
    with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5) as face_mesh:
 
        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
 
        if not results.multi_face_landmarks:
            print('Face not detected!!!')
            return 0
 
        for face_landmarks in results.multi_face_landmarks:
            values = np.array(face_landmarks.landmark)
            face_keypnts = np.zeros((len(values), 2))
 
            for idx,value in enumerate(values):
                face_keypnts[idx][0] = value.x
                face_keypnts[idx][1] = value.y
 
            # Convert normalized points to image coordinates
            face_keypnts = face_keypnts * (width, height)
            face_keypnts = face_keypnts.astype('int')
 
            relevant_keypnts = []
 
            for i in selected_keypoint_indices:
                relevant_keypnts.append(face_keypnts[i])
            return relevant_keypnts
    return 0

img1 = cv2.imread('/Users/kaiser/Documents/GitHub/fillter_lab/filter/test.jpg')
print(getLandmarks(img1))