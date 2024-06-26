from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from .config import MONGO_DB_NAME,MONGO_URL



# Get database URL from environment variable
DATABASE_URL =  MONGO_URL
DATABASE_NAME = MONGO_DB_NAME

# Connect to MongoDB
client = AsyncIOMotorClient(DATABASE_URL)
db = client[DATABASE_NAME]


# Access or create collections
User = db.get_collection("users")
Event = db.get_collection("events")
FaceImage=db.get_collection("face_images")


async def initialize_db():
    # Insert an initial document to create collections if they don't exist
    if not User.find_one():
        await User.insert_one({
            "_id": ObjectId(),
            "username": "testuser",
            "email": "testuser@example.com",
            "password": "password",
            "first_name": "Test",
            "last_name": "User",
            "phone_number": "1234567890",
            "birth_date": "2000-01-01T00:00:00Z",
            "account_type": "free",
            "is_verified": False
        })

    if not Event.find_one():
        await Event.insert_one({
            "_id": ObjectId(),
            "event_name": "Test Event",
            "event_url": "http://example.com/event",
            "created_at": "2024-05-23T00:00:00Z",
            "images_directory": "/path/to/images",
            "is_complete": False,
            "user_id": "testuser"
        })

    if not FaceImage.find_one():
        await FaceImage.insert_one({
            "_id": ObjectId(),
            "user_id": "testuser",
            "image_path1": "/path/to/image1.jpg",
            "image_path2": "/path/to/image2.jpg",
            "image_path3": "/path/to/image3.jpg"
        })
    print("database initialized sucessfully")






        


