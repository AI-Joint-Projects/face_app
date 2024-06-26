from fastapi import APIRouter, Depends, status, HTTPException
from fastapi.security.oauth2 import OAuth2PasswordRequestForm
from pydantic import EmailStr

from .. import database, schemas, models, utils, oauth2,otp

router = APIRouter(tags=['Authentication'])


@router.post('/login', response_model=schemas.Token)
async def login(user_credentials: OAuth2PasswordRequestForm = Depends()):

    user= await database.User.find_one({"email":user_credentials.username})
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=f"Invalid Credentials")
    user=models.user_serializer_auth(user)
    if not utils.verify(user_credentials.password, user["password"]):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=f"Invalid Credentials")
    access_token = oauth2.create_access_token(data={"user_id": user["id"]})

    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/forgot-password-generate")
async def forget_password_generate(email:EmailStr):
    collection= await database.User.find_one({"email":email})
    if collection is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,detail=f"the {email} doesn't exist")
    user= models.user_serializer(collection)
    otp_code= otp.generate_otp()
    await database.User.update_one({"email":email},{"$set":{"otp":otp_code}})
    otp.send_otp_email(email,otp_code)
    return {"message":"OTP sent successfully"}

@router.post("forget-password-verify")
async def forget_password_verify(user_otp:str,email:EmailStr):
    collection= await database.User.find_one({"email":email})
    vals=models.user_otp_serializer(collection)
    stored_otp= vals["otp"]
    if user_otp != stored_otp:
        raise HTTPException(status_code=status.HTTP_406_NOT_ACCEPTABLE,detail="incorrect otp")
    else:
        await database.User.update_one({"email":email},{"$set":{"is_verified":True}})
        return {"message": "successfully verified"}
    
@router.put("change-password")
async def change_password(email:EmailStr,new_password:str):
    hashed_password=utils.hash(new_password)
    await database.User.update_one({"email":email},{"$set":{"password":hashed_password}})
    return {"message":"password_changed"}





