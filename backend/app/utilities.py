import face_recognition
from sklearn import neighbors
import os 
import os.path
from face_recognition.face_recognition_cli import image_files_in_folder
import pickle
import random
import string
import cv2
import dlib
import math   
import numpy as np
import random 
from ultralytics import YOLO
from .config import SHAPE_PREDICTOR_PATH
from . import database


model= YOLO("./best.pt")
#utility functions here:
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}



face_images_collection = database.FaceImage

def train(model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    X = []
    y = []

    print("Successfully started")

    # Fetch all documents from the FaceImages collection
    face_images_docs = face_images_collection.find()

    for doc in face_images_docs:
        username = doc["username"]
        image_paths = [doc["image_path1"], doc["image_path2"], doc["image_path3"]]

        for img_path in image_paths:
            print(f"Processing image: {img_path}")
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(username)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    print("Model is trained")
    return knn_clf

#function to get the prediction from the trained knn
def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):
    
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:                          
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

#to generate event url
def generate_event_url(event_name: str) -> str:
    # Generate a random string of alphanumeric characters
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    
    # Replace spaces in the event name with hyphens and convert to lowercase
    event_name = event_name.replace(' ', '-').lower()
    
    # Combine the event name with the random string to create the URL
    event_url = f"{event_name}-{random_string}"
    
    return event_url


#code to extract three faces for comparison

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

def get_landmarks(image):
    dets = detector(image, 1)
    if len(dets) > 0:
        shape = predictor(image, dets[0])
        return shape
    return None

def calculate_eye_distance(landmarks):
    # Indices for left and right eyes
    left_eye_index = [36, 37, 38, 39, 40, 41]
    right_eye_index = [42, 43, 44, 45, 46, 47]

    # Extract coordinates of left and right eyes
    left_eye_pts = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in left_eye_index])
    right_eye_pts = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in right_eye_index])

    # Calculate distance between left and right eyes
    eye_distance = np.linalg.norm(left_eye_pts.mean(axis=0) - right_eye_pts.mean(axis=0))
    
    return eye_distance, left_eye_pts.mean(axis=0), right_eye_pts.mean(axis=0)


def delete_frames(input_video_path, percentage_to_delete=70):
    """Deletes a specified percentage of frames from a video while trying to preserve important information.

    Args:
        input_video_path (str): Path to the input video file.
        percentage_to_delete (int, optional): Percentage of frames to delete (0-100). Defaults to 70.
    """

    # Create a temporary path for the output video
    temp_output_path = input_video_path + ".temp.mp4"

    # Open the input video file
    cap = cv2.VideoCapture(input_video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the number of frames to delete
    frames_to_delete = int(total_frames * (percentage_to_delete / 100))

    # Initialize variables for output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(temp_output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    # Create a list of frame indices to keep
    keep_indices = list(range(total_frames))
    random.shuffle(keep_indices)  # Shuffle indices for randomness
    keep_indices = keep_indices[:total_frames - frames_to_delete]  # Keep the required number of frames

    # Process frames and write to output video
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if frame_count in keep_indices:
                output_video.write(frame)
            frame_count += 1
        else:
            break

    # Release video capture and writer
    cap.release()
    output_video.release()
    cv2.destroyAllWindows()

    # Delete the original video file
    os.remove(input_video_path)

    # Rename the temporary output video to the original path
    os.rename(temp_output_path, input_video_path)

    print(f"Successfully deleted {frames_to_delete} frames from '{input_video_path}' and replaced the original video.")

def process_video(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    min_eye_distance_left = float('inf')
    min_eye_distance_right = float('inf')
    max_eye_distance = float('-inf')
    max_frame = None
    left_min_frame = None
    right_min_frame = None
    initial_left_eye_coordinate = None
    initial_right_eye_coordinate = None
    
    frame_idx = 0
    first_frame = True

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        landmarks = get_landmarks(frame)
        if landmarks is not None:
            eye_distance, left_eye_coord, right_eye_coord = calculate_eye_distance(landmarks)
            
            # Save the maximum eye distance frame
            if eye_distance > max_eye_distance:
                max_eye_distance = eye_distance
                max_frame = frame

            if first_frame:
                initial_left_eye_coordinate = left_eye_coord
                initial_right_eye_coordinate = right_eye_coord
                first_frame = False
            else:
                if right_eye_coord[0] < initial_right_eye_coordinate[0]:  # Both eyes are left of the initial right eye coordinate
                    if eye_distance < min_eye_distance_left:
                        min_eye_distance_left = eye_distance
                        left_min_frame = frame
                if left_eye_coord[0] > initial_left_eye_coordinate[0]:  # Both eyes are right of the initial left eye coordinate
                    if eye_distance < min_eye_distance_right:
                        min_eye_distance_right = eye_distance
                        right_min_frame = frame
        
        print(f"Processing frame {frame_idx}/{total_frames} - Eye distance: {eye_distance if landmarks else 'N/A'}")
        frame_idx += 1

    cap.release()

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    frame_paths = []

    # Save frames
    if max_frame is not None:
        max_frame_path=os.path.join(output_dir, 'max_eye_distance_frame.jpg')
        cv2.imwrite(max_frame_path, max_frame)
        frame_paths.append(max_frame_path)
    if left_min_frame is not None:
        left_min_frame_path=os.path.join(output_dir, 'left_min_eye_distance_frame.jpg')
        cv2.imwrite(left_min_frame_path, left_min_frame)
        frame_paths.append(left_min_frame_path)
    if right_min_frame is not None:
        right_min_frame_path=os.path.join(output_dir, 'right_min_eye_distance_frame.jpg')
        cv2.imwrite(right_min_frame_path, right_min_frame)
        frame_paths.append(right_min_frame_path)
    
    print(f"Frames saved in {output_dir}")
    
    return frame_paths

def get_faces(image_path, conf_threshold=0.8):
    results = model(image_path, conf=conf_threshold)
    bounding_boxes = results[0].boxes.xyxy.cpu().numpy()  # Convert to numpy array
    confidences = results[0].boxes.conf.cpu().numpy()  # Extract confidence scores
    return bounding_boxes, confidences

# Function to crop the highest confidence bounding box
def crop_highest_conf_box(image_path, bounding_boxes, confidences):
    image = cv2.imread(image_path)
    
    # Find the bounding box with the highest confidence
    if len(confidences) > 0:
        max_conf_idx = confidences.argmax()
        highest_conf_box = bounding_boxes[max_conf_idx]
        
        # Crop the image to the bounding box
        x_min, y_min, x_max, y_max = map(int, highest_conf_box)
        cropped_face = image[y_min:y_max, x_min:x_max]
        
        return cropped_face
    else:
        raise ValueError("No bounding boxes detected with the given confidence threshold.")
    
def process_images(image_path):
    bounding_boxes, confidences = get_faces(image_path)
    try:
        # Crop the face with the highest confidence
        cropped_face = crop_highest_conf_box(image_path, bounding_boxes, confidences)
    except ValueError as e:
        print(e)
        return

    # Save the cropped face to a temporary path
    save_dir = os.path.dirname(image_path)
    temp_save_path = os.path.join(save_dir, "temp_aligned.jpg")
    cv2.imwrite(temp_save_path, cropped_face)
    print(f"Saved aligned face image to {temp_save_path}")

    # Delete the original image
    os.remove(image_path)

    # Rename the temporary image to the original path
    os.rename(temp_save_path, image_path)

    print(f"Replaced original image with aligned face image at {image_path}")
