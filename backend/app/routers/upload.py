from fastapi import APIRouter,Depends,status, File, UploadFile,HTTPException,Body
from .. import models,database,oauth2
from typing import List
import os
import shutil

router=APIRouter(
    prefix="/upload-images"
)

@router.put("/local-storage", status_code=status.HTTP_202_ACCEPTED)
async def local_storage(
    event: str,
    files: List[UploadFile] = File(...),
    current_user: models.UserModel = Depends(oauth2.get_current_user)
):
    user = await database.User.find_one({"_id": current_user["_id"]})
    username = models.user_serializer(user)["username"]

    # Check if the event exists
    event_data = await database.Event.find_one({"event_url": event}) 
    if not event_data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"The given event: {event} doesn't exist")

    event_data = models.event_serializer(event_data)
    if event_data["username"] != username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="You don't have access to modify the event")

    # Create directory for storing images
    base_dir = "event_images"
    event_dir = os.path.join(base_dir, event)
    os.makedirs(event_dir, exist_ok=True)

    # Save files in the event directory with original file extension
    for idx, file in enumerate(files, start=1):
        file_extension = os.path.splitext(file.filename)[1] # Get file extension from filename
        file_path = os.path.join(event_dir, f"{idx}{file_extension}")
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to save file {file.filename}: {str(e)}")

    # Store the path of the directory in the database
    await database.Event.update_one({"event_url": event}, {"$set": {"image_directory": event_dir}})

    return {"message": f"Files successfully saved in {event_dir}", "directory": event_dir}


