from io import BytesIO
import io
import os
import shutil
from uuid import uuid4
import zipfile
import cv2
from fastapi import APIRouter, File, UploadFile, status, Body, HTTPException,Depends
from fastapi.responses import JSONResponse, StreamingResponse
from .. import models, database,oauth2,utilities,schemas

router = APIRouter(
    prefix="/profile",
    tags=["Profile"]
)

@router.delete("/delete-user",status_code=status.HTTP_200_OK)
async def delete_user( current_user: dict = Depends(oauth2.get_current_user)):
    username=models.user_serializer(await database.User.find_one({"_id":current_user["_id"]}))["username"]
    await database.Event.update_many({"username":username},{"$pull":{"joined_by":{"username":username}}})
    delete= await database.User.delete_one({"_id":current_user["_id"]})
    if delete.acknowledged:
        return {"message":"user: {username} account has been deleted sucessfully"}
    else: 
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,detail="database opration failed")


@router.delete("/delete-event",status_code=status.HTTP_200_OK)
async def delete_event(event_url:str ,current_user:models.UserModel=Depends(oauth2.get_current_user)):
    user = await database.User.find_one({"_id": current_user["_id"]})
    username = models.user_serializer(user)["username"]

    # Check if the event exists
    event_data = await database.Event.find_one({"event_url": event_url}) 
    if not event_data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"The given event: {event_url} doesn't exist")
    event_data = models.event_serializer(event_data)
    if event_data["username"] != username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="You don't have access to modify the event")
    await database.Event.delete_one({"event_url":event_url})
    await database.User.update_one({"_id":current_user["_id"]},{"$pull": {"joined_events": {"event_url": event_url}}})
    return {"message":f"{event_url} is deleted"}


@router.put("/leave-event",status_code=status.HTTP_200_OK)
async def leave_event(event_id:str,current_user:models.UserModel=Depends(oauth2.get_current_user)):
    username=models.user_serializer(await database.User.find_one({"_id":current_user["_id"]}))
    up1=await database.Event.update_many({"username":username},{"$pull":{"joined_by":{"username":username}}})
    up2=await database.User.update_one({"_id":current_user["_id"]},{"$pull": {"joined_events": {"event_url": event_id}}})
    if up1.acknowledged and up2.acknowledged:
        return {"message":f"event :{event_id} left successfully"}
    else:
        return {"message":f"leaving event failed"}



@router.put("/update-video",status_code=status.HTTP_202_ACCEPTED)
async def update_video(video: UploadFile = File(...),current_user:models.UserModel=Depends(oauth2.get_current_user)):
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

        # Delete the original video file
        os.remove(video_path)

        # Process the images 
        for item in frame_paths:
            try:
                utilities.process_images(item)
            except Exception as img_err:
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(img_err))

        try:
            # Save to MongoDB
            await database.FaceImage.update_one({"username":username},{"$set",{"image_path1":frame_paths[0],"image_path2":frame_paths[1],"image_path3":frame_paths[3]}})

            return {"processed_frames_directory": output_dir, "frame_paths": frame_paths}
        except Exception as db_err:
            return JSONResponse(status_code=500, content={"message": "An error occurred on await database", "error": str(db_err)})

    except Exception as vid_err:
        return JSONResponse(status_code=500, content={"message": "An error occurred while processing the video.", "error": str(vid_err)})
    
@router.get("/get-username",status_code=status.HTTP_200_OK)
async def get_username(current_user:models.UserModel=Depends(oauth2.get_current_user)):
    username = models.user_serializer(await database.User.find_one({"_id": current_user["_id"]}))["username"]
    return {"username":username}

@router.put("/upload-profile-picture", status_code=status.HTTP_202_ACCEPTED)
async def upload_profile(image: UploadFile = File(...), current_user: models.UserModel = Depends(oauth2.get_current_user)):
    username = models.user_serializer(await database.User.find_one({"_id": current_user["_id"]}))["username"]
    base_images_dir = "profile-images"
    user_images_dir = os.path.join(base_images_dir, username)
    os.makedirs(user_images_dir, exist_ok=True)

    file_extension = os.path.splitext(image.filename)[1]  # Get the file extension
    image_path = os.path.join(user_images_dir, f"profile_picture{file_extension}")

    # Save the uploaded image file
    with open(image_path, "wb") as f:
        f.write(await image.read())

    # Update the user's profile picture path in the await database
    await database.User.update_one({"_id": current_user["_id"]}, {"$set": {"profile_picture": image_path}})

    return {"username": username, "profile_picture": image_path}

@router.get("/profile-picture", status_code=status.HTTP_200_OK)
async def get_profile_picture(current_user: models.UserModel = Depends(oauth2.get_current_user)):
    user_doc = await database.User.find_one({"_id": current_user["_id"]})
    
    if user_doc is None:
        raise HTTPException(status_code=404, detail="User not found")

    profile_picture_path = user_doc.get("profile_picture")
    
    if not profile_picture_path or not os.path.exists(profile_picture_path):
        raise HTTPException(status_code=404, detail="Profile picture not found")

    return stream_profile_picture_as_zip(profile_picture_path)

def stream_profile_picture_as_zip(profile_picture_path: str):
    def zip_generator():
        with io.BytesIO() as buffer:
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                with open(profile_picture_path, "rb") as f:
                    zip_file.writestr(os.path.basename(profile_picture_path), f.read())
            buffer.seek(0)
            yield from buffer

    headers = {
        "Content-Disposition": f"attachment; filename=profile_picture.zip",
        "Content-Type": "application/zip"
    }

    return StreamingResponse(zip_generator(), headers=headers)

@router.get("/get-faces", response_class=StreamingResponse)
async def get_faces(current_user: models.UserModel = Depends(oauth2.get_current_user)):
    user_doc = await database.User.find_one({"_id": current_user["_id"]})
    
    if user_doc is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    username = user_doc["username"]
    face_images_doc = await database.FaceImage.find_one({"username": username})
    
    if not face_images_doc:
        raise HTTPException(status_code=404, detail="Face images not found")
    
    image_paths = [
        face_images_doc["image_path1"],
        face_images_doc["image_path2"],
        face_images_doc["image_path3"]
    ]
    
    # Check if all image paths exist
    for image_path in image_paths:
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")
    
     # Create a zip file in memory
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for image_path in image_paths:
            zip_file.write(image_path, os.path.basename(image_path))
    zip_buffer.seek(0)

    return StreamingResponse(zip_buffer, media_type="application/x-zip-compressed", headers={"Content-Disposition": "attachment;filename=profile_pictures.zip"})

@router.get("/joined-count", status_code=status.HTTP_200_OK)
async def count_joined_events(current_user: models.UserModel = Depends(oauth2.get_current_user)):
    user_doc = await database.User.find_one({"_id": current_user["_id"]})
    
    if user_doc is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    joined_events = user_doc.get("joined_events", [])
    joined_events_count = len(joined_events)
    
    return {"username": user_doc["username"], "joined_events_count": joined_events_count}
        

@router.get("/created-count", status_code=status.HTTP_200_OK)
async def count_created_events(current_user: models.UserModel = Depends(oauth2.get_current_user)):
    collection=await database.User.find_one({"_id": current_user["_id"]})
    username = models.user_serializer(collection)["username"]
    # Count the number of events created by the specified username
    event_count = await database.Event.count_documents({"username": username})
    
    return {"username": username, "created_events_count": event_count}