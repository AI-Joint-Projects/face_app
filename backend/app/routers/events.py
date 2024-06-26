import io
from typing import List
import zipfile
from fastapi import APIRouter, status, Body, HTTPException,Depends
from fastapi.responses import StreamingResponse
from .. import models, database,oauth2,utilities,schemas,utils
from bson import ObjectId
from datetime import datetime
import os

router = APIRouter(
    prefix="/events"
)

@router.post("/create-event", status_code=status.HTTP_201_CREATED)
async def create_event(event: models.EventModel = Body(...), current_user: models.UserModel = Depends(oauth2.get_current_user)):
    event_data = event.model_dump(exclude=["id", "username", "event_url", "created_at"])
    created_at = datetime.now()
    user = await database.User.find_one({"_id": current_user["_id"]})
    doc=models.user_serializer(user)
    if doc["is_verified"]==True:

        username = doc["username"]
        event_link = utilities.generate_event_url(event_data["event_name"])
        event_data.update({"event_url": event_link, "username": username, "datetime": created_at})
        new_event = await database.Event.insert_one(event_data)
        if not new_event.acknowledged:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Event creation failed")
        
        # Create directory path
        base_dir = "event_images"
        event_dir = os.path.join(base_dir, event_link)

        # Ensure the directory exists
        os.makedirs(event_dir, exist_ok=True)

        # Update event data with directory path
        await database.Event.update_one({"_id": ObjectId(new_event.inserted_id)}, {"$set": {"images_directory": event_dir}})

        created_event =await database.Event.find_one({"_id": ObjectId(new_event.inserted_id)})
        if created_event is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Something is wrong with the created event query")

        response = models.event_serializer_2(created_event)
        return {"event_link": response["event_url"]}
    else:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,status="Please complete the otp verification")



@router.get("/created-events", status_code=status.HTTP_200_OK)
async def created_events(current_user: models.UserModel = Depends(oauth2.get_current_user)):
    user = await database.User.find_one({"_id": current_user["_id"]})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    doc= models.user_serializer(user)
    if doc["is_verified"]==True:
        username =doc["username"]
        events_cursor = database.Event.find({"username": username})
        events = await events_cursor.to_list(length=None)  # Retrieve all events as a list
        
        serialized_events = models.events_serializer(events)
        
        return {"created_events": serialized_events}
    else: 
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,detail="Please complete the otp verification")


@router.put("/join-event", status_code=status.HTTP_202_ACCEPTED)
async def join_event(event: schemas.JoinEvent = Body(...), current_user: models.UserModel = Depends(oauth2.get_current_user)): 
    user = await database.User.find_one({"_id": current_user["_id"]})
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    
    doc= models.user_serializer(user)
    if doc["is_verified"]==True:
        username = doc["username"]

        event_doc = await database.Event.find_one({"event_url": event.url})
        if not event_doc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"The given event: {event.url} doesn't exist")

        try: 
            # Add the username to the 'joined_by' array in the event document
            await database.Event.update_one(
                {"event_url": event.url},
                {"$addToSet": {"joined_by": {"username": username}}}
            )

            # Add the event URL to the 'joined_events' array in the user document
            await database.User.update_one(
                {"_id": current_user["_id"]},
                {"$addToSet": {"joined_events": {"event_url": event.url}}}
            )
        except Exception as e:
            # Fallback to $push if $addToSet fails
            await database.Event.update_one(
                {"event_url": event.url},
                {"$push": {"joined_by": {"username": username}}}
            )

            await database.User.update_one(
                {"_id": current_user["_id"]},
                {"$push": {"joined_events": {"event_url": event.url}}}
            )

        return {"message": f"Event joined: {event.url}"}
    else: 
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,detail="Please complete the otp verification")



@router.get("/joined-events", status_code=status.HTTP_200_OK)
async def get_joined_events(current_user: models.UserModel = Depends(oauth2.get_current_user)):
    # Find the user document
    user = await database.User.find_one({"_id": current_user["_id"]})
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    
    user = models.user_event_serializer(user)
    if user["is_verified"]==True:
    
        # Get the joined_events attribute, defaulting to an empty list if not present
        events_joined = user.get("joined_events", [])
        
        # Initialize an empty list to store event details
        events_details = []

        # Iterate through each joined event
        for event in events_joined:
            event_url = event["event_url"]
            event_doc = await database.Event.find_one({"event_url": event_url})
            
            if event_doc:
                event_doc = models.event_serializer(event_doc)
                # Append the event details to the list
                event_detail = {
                    "event_name": event_doc["event_name"],
                    "event_url": event_doc["event_url"],
                    "is_complete": event_doc["is_complete"]
                }
                events_details.append(event_detail)
        
        return {"joined_events": events_details}
    else:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,detail="Please complete the otp verification")



@router.put("/complete-event", status_code=status.HTTP_200_OK)
async def complete_event(complete: bool, event_url: str, current_user: models.UserModel = Depends(oauth2.get_current_user)):
    user = await database.User.find_one({"_id": current_user["_id"]})
    doc=models.user_serializer(user)
    username = doc["username"]
    if doc["is_verified"]==True:

        # Check if the event exists
        event_data = await database.Event.find_one({"event_url": event_url})
        if not event_data:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"The given event: {event_url} doesn't exist")

        event_data = models.event_serializer_3(event_data)
        if event_data["username"] != username:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="You don't have access to modify the event")

        if complete:
            # Retrieve the images directory path from the event document
            path = event_data.get("images_directory")
            if not path:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"The images directory for the event: {event_url} is not specified")
            
            # Properly format the path for the current operating system
            path = os.path.normpath(path)

            if not os.path.isdir(path):
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"The images directory for the event: {event_url} at path: {path} is invalid or does not exist")
            try:
                # Train the KNN model
                knn =await utils.train(event_url=event_url)
            except ValueError as e:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

            stores = {}

            # Process each image in the event directory
            try:
                for img_name in os.listdir(path):
                    img_path = os.path.join(path, img_name)
                    if not os.path.isfile(img_path):
                        continue
                    
                    identities = utils.prediction(img_path, knn_clf=knn)
                    stores[img_path] = identities
            except FileNotFoundError as e:
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"File not found error: {str(e)}")
            except Exception as e:
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An error occurred while processing images: {str(e)}")
            
            # Update each user's document in the database
            joined_by_list= event_data["joined_by"]
            print("working till now")
            print(joined_by_list)
            for joined_user in joined_by_list:
                joined_username = joined_user["username"]
                user = await database.User.find_one({"username": joined_username})
                if user:
                    user_event_images = user.get("event_images", [])
                    # Check if the user already has the event in their event_images
                    event_exists = False
                    for event in user_event_images:
                        if event["event_url"] == event_url:
                            event["images"] = list(stores.keys())
                            event_exists = True
                            break
                    
                    if not event_exists:
                        user_event_images.append({
                            "event_url": event_url,
                            "images": list(stores.keys())
                        })

                    # Log the update operation's acknowledgment status
                    update_result = await database.User.update_one(
                        {"username": joined_username},
                        {"$set": {"event_images": user_event_images}}
                    )
                    if not update_result.acknowledged:
                        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update user document with event images")

            # Mark the event as complete
            await database.Event.update_one({"event_url": event_url}, {"$set": {"is_complete": complete}})

            return {"message": "Event marked as complete and images processed", "stores": stores}
        else:
            return {"message": "Event not marked as complete"}
    else:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,detail="OTP not verified")


            

@router.get("/event-images", status_code=status.HTTP_200_OK)
async def get_event_images(event_url: str, current_user: models.UserModel = Depends(oauth2.get_current_user)):
    # Find the user document
    user =await database.User.find_one({"_id": current_user["_id"]})
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    
    user_data = models.user_image_serializer(user)
    if user_data["is_verified"]==True:
        # Find the event in the user's event_images
        event_images_list = user_data["event_images"]
        for event_images in event_images_list:
            if event_images["event_url"] == event_url:
                image_paths = event_images["images"]
                return stream_images_as_zip(image_paths, event_url)
        
        # If the event is not found in the user's event_images
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No images found for the event: {event_url}")
    else: 
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)

def stream_images_as_zip(image_paths: List[str], event_url: str):
    def zip_generator():
        with io.BytesIO() as buffer:
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for image_path in image_paths:
                    if os.path.exists(image_path):
                        with open(image_path, "rb") as f:
                            zip_file.writestr(os.path.basename(image_path), f.read())
                    else:
                        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Image {image_path} not found")
            buffer.seek(0)
            yield from buffer
    
    headers = {
        "Content-Disposition": f"attachment; filename={event_url}.zip",
        "Content-Type": "application/zip"
    }
    
    return StreamingResponse(zip_generator(), headers=headers)




        





    

    

