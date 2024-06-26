from email.mime.text import MIMEText
from fastapi import APIRouter, status, Body, HTTPException,Depends,File,UploadFile
from fastapi.responses import JSONResponse
from .. import models,database,utils,oauth2,utilities,otp
import os
import shutil
from uuid import uuid4
import cv2
import aiofiles
import math
import random
import smtplib

router = APIRouter(
    prefix="/users",
    tags=["users"]
)

@router.post("/register", status_code=status.HTTP_201_CREATED)
async def create_user(user: models.UserModel = Body(...)):  
    user_data = user.model_dump(by_alias=True, exclude=["id"])
    user_data["password"]=utils.hash(user_data["password"])
    # Check if username or email already exists
    username_exist = await database.User.find_one({"username": user.username})
    email_exist = await database.User.find_one({"email": user.email})

    if username_exist:
        raise HTTPException(
            status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
            detail=f"Account with username '{user.username}' already exists"
        )
    if email_exist:
        raise HTTPException(
            status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
            detail=f"Account with email '{user.email}' already exists"
        )
 
    # Insert new user
    new_user = await database.User.insert_one(user_data)
    created_user = await database.User.find_one({"_id": new_user.inserted_id})

    # Serialize the user data for response
    response = models.user_serializer(created_user)
    return response

@router.post("/generate-otp")
async def generate_otp(current_user: models.UserModel = Depends(oauth2.get_current_user)):
    otp_code= otp.generate_otp()
    await database.User.update_one({"_id":current_user["_id"]},{"$set":{"otp":otp_code}})
    email= models.user_serializer(await database.User.find_one({"_id":current_user["_id"]}))["email"]
    otp.send_otp_email(email,otp_code)
    return {"message":"OTP sent successfully"}

@router.post("/verify-otp",status_code= status.HTTP_202_ACCEPTED)
async def verify_otp(user_otp:str,current_user: models.UserModel = Depends(oauth2.get_current_user)):
    collection= await database.User.find_one({"_id":current_user["_id"]})
    vals=models.user_otp_serializer(collection)
    stored_otp= vals["otp"]
    if user_otp != stored_otp:
        raise HTTPException(status_code=status.HTTP_406_NOT_ACCEPTABLE,detail="incorrect otp")
    else:
        await database.User.update_one({"_id":current_user["_id"]},{"$set":{"is_verified":True}})
        return {"message": "successfully verified"}

@router.post("/take-images",status_code=status.HTTP_202_ACCEPTED)    
async def take_images(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    file3: UploadFile = File(...),
    current_user: dict = Depends(oauth2.get_current_user)
):
    username=models.user_serializer(await database.User.find_one({"_id":current_user["_id"]}))["username"]
    base_images_dir = "images"
    user_images_dir = os.path.join(base_images_dir, str(username))
    os.makedirs(user_images_dir, exist_ok=True)

    try:
        face_image_paths = []
        for idx, file in enumerate([file1, file2, file3], start=1):
            file_location = os.path.join(user_images_dir, f"image_{idx}_{uuid4()}.jpg")
            with open(file_location, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Load the image with OpenCV
            image = cv2.imread(file_location)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Load OpenCV's Haar cascade for face detection
            haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Detect faces
            faces = haar_cascade.detectMultiScale(gray_image, scaleFactor=1.4, minNeighbors=5)

            if len(faces) == 0:
                os.remove(file_location)
                return JSONResponse(status_code=400, content={"message": f"No face detected in image {idx}, please upload images with faces."})

            # Save only the first detected face
            x, y, w, h = faces[0]
            face_image = image[y:y+h, x:x+w]
            resized = cv2.resize(face_image, (128, 128))
            face_image_path = os.path.join(user_images_dir, f"face_{idx}_{uuid4()}.jpg")
            cv2.imwrite(face_image_path, resized)
            face_image_paths.append(face_image_path)

            # Optionally delete the original image
            os.remove(file_location)

        # Create a FaceImages instance
        face_images_record = models.FaceImages(
            username=username,
            image_path1=face_image_paths[0],
            image_path2=face_image_paths[1],
            image_path3=face_image_paths[2]
        )

        # Save to MongoDB
        await database.FaceImage.insert_one(face_images_record.model_dump(by_alias=True))

        return {"faces_directory": user_images_dir, "image_paths": face_image_paths}

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": "An error occurred while processing the images.", "error": str(e)})

@router.post("/process-video", status_code=status.HTTP_202_ACCEPTED)
async def process_video(
    video: UploadFile = File(...),
    current_user: dict = Depends(oauth2.get_current_user)
):
    username = models.user_serializer(await database.User.find_one({"_id": current_user["_id"]}))["username"]
    base_videos_dir = "videos"
    user_videos_dir = os.path.join(base_videos_dir, str(username))
    os.makedirs(user_videos_dir, exist_ok=True)

    try:
        # Save the uploaded video
        video_path = os.path.join(user_videos_dir, f"video_{uuid4()}.mp4")
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # Check the duration of the video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        cap.release()

        if duration >= 7:
            os.remove(video_path)
            return JSONResponse(status_code=400, content={"message": "Video too long. Please upload a video shorter than 5 seconds."})

        # Define output directory for frames
        output_dir = os.path.join(user_videos_dir, "processed_frames")

        # Delete frames to speed up processing
        utilities.delete_frames(video_path, percentage_to_delete=70)

        # Process the video and get frame paths
        frame_paths = utilities.process_video(video_path, output_dir)

        # Check if all required frames are found
        if len(frame_paths) < 3:
            os.remove(video_path)
            return JSONResponse(status_code=400, content={"message": "Unable to extract all required frames from the video."})

        

        # Process the images 
        for item in frame_paths:
            try:
                utilities.process_images(item)
            except Exception as img_err:
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(img_err))

        try:
            # Create a FaceImages instance
            face_images_record = models.FaceImages(
                username=username,
                image_path1=frame_paths[0],
                image_path2=frame_paths[1],
                image_path3=frame_paths[2]
            )

            # Save to MongoDB                                   
            await database.FaceImage.insert_one(face_images_record.model_dump(by_alias=True))

            return {"processed_frames_directory": output_dir, "frame_paths": frame_paths}
        except Exception as db_err:
            return JSONResponse(status_code=500, content={"message": "An error occurred on await database", "error": str(db_err)})

    except Exception as vid_err:
        return JSONResponse(status_code=500, content={"message": "An error occurred while processing the video.", "error": str(vid_err)})



    



