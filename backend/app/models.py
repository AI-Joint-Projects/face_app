from pydantic import BaseModel,EmailStr,Field,ConfigDict
from typing import Optional,List,Any
from pydantic.functional_validators import BeforeValidator
from typing_extensions import Annotated
from datetime import datetime
from bson import ObjectId

PyObjectId = Annotated[str, BeforeValidator(str)]

class UserModel(BaseModel):
    id: Optional[PyObjectId]=Field(alias="_id",default=None)
    first_name: str= Field(...)
    last_name: str=Field(...)
    username: str=Field(...)
    email: EmailStr=Field(...)
    password: str= Field(...)
    birthdate:datetime = Field(..., description="The user's birthdate in YYYY-MM-DD format")
    is_verified: Optional[bool]=False
    account_type:Optional[str] = "basic"

    model_config=ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed= True,
        json_schema_extra={
            "example":{
                "first_name":"Jane",
                "last_name":"Doe",
                "email":"jdoe@example.com",
                "username":"test",
                "password":"string",
                "birthdate":"2022-01-01",
                "account_type":"basic",
                "is_verified":False
            }
            },
    )
def user_serializer(user: dict) -> dict:
    return {
        "id": str(user["_id"]),
        "username": user["username"],
        "email": user["email"],
        "first_name": user["first_name"],
        "last_name": user["last_name"],
        "birthdate": user.get("birthdate"),
        "account_type": user["account_type"],
        "is_verified": user["is_verified"]
    }
def user_serializer_auth(user: dict) -> dict:
    return {
        "id": str(user["_id"]),
        "username": user["username"],
        "email": user["email"],
        "first_name": user["first_name"],
        "last_name": user["last_name"],
        "birthdate": user.get("birthdate"),
        "account_type": user["account_type"],
        "password":user["password"],
        "is_verified":user["is_verified"]
    }
def user_event_serializer(user: dict) -> dict:
    return {
        "id": str(user["_id"]),
        "username": user["username"],
        "email": user["email"],
        "first_name": user["first_name"],
        "last_name": user["last_name"],
        "birthdate": user.get("birthdate"),
        "account_type": user["account_type"],
        "is_verified":user["is_verified"],
        "joined_events": user.get("joined_events", [])
    }
def user_image_serializer(user:dict) -> dict:
    return {
        "id": str(user["_id"]),
        "username": user["username"],
        "email": user["email"],
        "first_name": user["first_name"],
        "last_name": user["last_name"],
        "birthdate": user.get("birthdate"),
        "account_type": user["account_type"],
        "is_verified":user["is_verified"],
        "joined_events": user.get("joined_events", []),
        "event_images":user.get("event_images",[])
    }


def user_otp_serializer(user: dict) -> dict:
    return {
        "is_verified": user["is_verified"],
        "otp":user["otp"]
    }


class EventModel(BaseModel):
    id: Optional[PyObjectId]=Field(alias="_id",default=None)
    event_name: str=Field(...)
    event_url: Optional[str]
    created_at: datetime = datetime.now()
    images_directory: Optional[str] = None
    is_complete: bool = False
    username: Optional[str]

    class Config:
        populate_by_field_name = True
        json_schema_extra = {
            "example": {
                "event_name": "example_event",
                "id":"str",
                "event_url":"str",
                "created_at":datetime.now(),
                "images_directory":"str",
                "username":"str",
                "is_complete": False
            }
        }
def event_serializer(event)-> dict:
    return{
        "event_name": event["event_name"],
        "event_url": event["event_url"],
        "created_at": event["datetime"],
        "images_directory": event["images_directory"],
        "is_complete": event["is_complete"],
        "username": event["username"]
    }
def event_serializer_2(event)-> dict:
    return{
        "event_name": event["event_name"],
        "event_url": event["event_url"],
        "images_directory": event["images_directory"],
        "is_complete": event["is_complete"],
        "username": event["username"]
    }

def event_serializer_3(event)-> dict:
    return{
        "event_name": event["event_name"],
        "event_url": event["event_url"],
        "images_directory": event["images_directory"],
        "is_complete": event["is_complete"],
        "username": event["username"],
        "joined_by":event.get("joined_by",[])
    }
def events_serializer(events) -> list:
    return [event_serializer(event) for event in events]


class FaceImages(BaseModel):
    username: str
    image_path1: str
    image_path2: str
    image_path3: str

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}