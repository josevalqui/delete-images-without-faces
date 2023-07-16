import os
from mtcnn import MTCNN
import cv2

# Initialize the MTCNN model
mtcnn = MTCNN(min_face_size=160, 
              steps_threshold = [0.8, 0.8, 0.8]
            )

# Go over all directories in the current directory
for dirpath, dirnames, filenames in os.walk('.'):
    for filename in filenames:
        if filename.endswith('.jpg') or filename.endswith('.png'): # check if file is an image
            # Full path to the file
            file_path = os.path.join(dirpath, filename)
            # Load image
            img = cv2.imread(file_path)
            if img is not None:
                # Convert color style from BGR to RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Perform face detection
                faces = mtcnn.detect_faces(img_rgb)
                # If no faces are detected, remove the image
                if len(faces) == 0:
                    os.remove(file_path)
                    print(f'Removed: {file_path}')