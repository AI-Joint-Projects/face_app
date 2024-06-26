from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    id:str

class CreateEvent(BaseModel):
    event_name: str

class JoinEvent(BaseModel):
    url: str