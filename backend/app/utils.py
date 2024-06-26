from passlib.context import CryptContext
import os
import pickle
import numpy as np
from sklearn import neighbors
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
import math   
from . import database
from ultralytics import YOLO
import cv2
from PIL import Image
from .config import EMBEDDING_MODEL_PATH,BEST_MODEL_PATH



pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash(password: str):
    return pwd_context.hash(password)


def verify(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


#for face recognition with own model 

# Load the YOLO model
model = YOLO(BEST_MODEL_PATH)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg',"webp"}

# Load the embedding model
embedding_model_path = EMBEDDING_MODEL_PATH
embedding_model = load_model(embedding_model_path)

# Database connection
face_images_collection = database.FaceImage
event_collection= database.Event

def extract_faces(image_path, detector):
    image = cv2.imread(image_path)
    results = detector(image)
    faces = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Extract bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            face = image[y1:y2, x1:x2]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            faces.append(Image.fromarray(face))
    return faces

def extract_embedding(image, embedding_model):
    image = image.resize((200, 200))  # Ensure the image is the correct size
    img_array = img_to_array(image) / 255.0
    embedding = embedding_model.predict(np.expand_dims(img_array, axis=0))[0]
    return embedding


async def train(model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False, event_url=None):
    X = []
    y = []

    print("Successfully started")

    if event_url is None:
        raise ValueError("Event URL must be provided")

    # Fetch the event document based on the event_url
    event_doc = await event_collection.find_one({"event_url": event_url})
    if not event_doc:
        raise ValueError(f"No event found with URL: {event_url}")

    # Get the list of usernames who joined the event
    joined_by_list = event_doc.get('joined_by', [])
    if not joined_by_list:
        raise ValueError(f"No users found for event with URL: {event_url}")

    # Extract usernames from the joined_by list
    usernames = [user["username"] for user in joined_by_list]

    # Fetch all face images for users who joined the event
    face_images_cursor = face_images_collection.find({"username": {"$in": usernames}})
    
    async for doc in face_images_cursor:
        username = doc["username"]
        image_paths = [doc["image_path1"], doc["image_path2"], doc["image_path3"]]

        for img_path in image_paths:
            print(f"Processing image: {img_path}")
            image = load_img(img_path, target_size=(200, 200))

            # Add face encoding for current image to the training set
            X.append(extract_embedding(image, embedding_model))
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


def prediction(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        print("on this ")
        raise Exception("Invalid image path: {}".format(X_img_path))
    print("after this")

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either through knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    faces = extract_faces(X_img_path, model)
    identities = []

    for face in faces:
        embedding = extract_embedding(face, embedding_model)

        # Calculate distances to nearest neighbors
        distances, indices = knn_clf.kneighbors([embedding], n_neighbors=3)

        # Calculate the average distance
        avg_distance = np.mean(distances[0])

        # Check if the average distance is above the threshold
        if avg_distance > distance_threshold:
            identity = "unknown"
        else:
            # Use majority voting for the class prediction
            neighbor_labels = knn_clf.predict([embedding])
            identity = max(set(neighbor_labels), key=list(neighbor_labels).count)

        identities.append(identity)

    return identities
