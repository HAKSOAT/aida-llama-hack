from pydantic import BaseModel
from typing import Literal

class BaseResponse(BaseModel):
    result: str

class CrisisType(BaseResponse):
    pass

class Caption(BaseResponse):
    result: dict[Literal["title", "description"], str]


class AidTags(BaseModel):
    result: dict[str, int]

class Commentary(BaseResponse):
    pass

class RealtimeDescription(BaseResponse):
    pass

class Segmentation(BaseResponse):
    pass
