from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import Response
from PIL import Image
import matplotlib.pyplot as plt

from backend.apis import mock_images
from backend.apis.llama import (
    LlamaAidTagging,
    LlamaCaption,
    LlamaClassification,
    LlamaRealtimeDescription,
    SAM_segment
)
from backend.schemas.response_schema import CrisisType, Caption, AidTags, Commentary, RealtimeDescription, Segmentation
from .fake_db import db
import requests
from io import BytesIO

router = APIRouter()
logger = logging.getLogger(__name__)

def open_image_from_url(url):
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for HTTP issues
    image = Image.open(BytesIO(response.content))
    return image

@router.get("/")
async def index() -> dict[str, str]:
    return {
        "info": "This is the index page of fastapi-nano. "
        "You probably want to go to 'http://<hostname:port>/docs'.",
    }


@router.get("/get-crisis-type")
async def get_crisis_type(
    request: Request,
    # auth: Depends = Depends(get_current_user)
    event_id: str = "valencia-flood",
    response_model=CrisisType
) -> CrisisType:

    data = [x for x in db if x["id"] == event_id][0]
    before_after = data["before_after"]
    images = [open_image_from_url(x) for x in before_after]
    model = request.app.state.llama_model
    llama_classification = LlamaClassification(model)
    result = llama_classification.image_to_text(images)
    text = result[0]
    return CrisisType(result=text)


@router.get("/get-caption")
async def get_caption(request: Request, crisis_type: str,
                      event_id: str = "valencia-flood",
                      response_model=Caption) -> Caption:
    data = [x for x in db if x["id"] == event_id][0]
    before_after = data["user_images"]
    images = [open_image_from_url(x) for x in before_after]

    model = request.app.state.llama_model
    llama_caption = LlamaCaption(crisis_type, model)
    result = llama_caption.image_to_text(images)
    obj = {k.lower(): v for k, v in json.loads(result[0]).items()}
    return Caption(result=obj)



@router.get("/get-description")
async def get_description(request: Request, crisis_type: str, event_id: str = "valencia-flood",
                          response_model=RealtimeDescription,) -> RealtimeDescription:
    model = request.app.state.llama_model
    llama_realtime_description = LlamaRealtimeDescription(crisis_type, model)
    data = [x for x in db if x["id"] == event_id][0]
    commentary = data["user_comments"]
    images = data["user_images"]
    
    img_captions = []
    llama_caption = LlamaCaption(crisis_type, model)
    for im in images:
        opened_image = open_image_from_url(im)
        result = llama_caption.image_to_text([opened_image])
        img_captions.append(result)
        
    print("image captions:", img_captions)
    
    result = llama_realtime_description.custom_inference(commentary, img_captions)
    return RealtimeDescription(result=result[0])


@router.get("/get-aid-tags")
async def get_aid_tags(request: Request, crisis_type: str, event_id: str = "valencia-flood",
                       response_model=AidTags) -> AidTags:

    texts = [x for x in db if x["id"] == event_id][0]

    model = request.app.state.llama_model
    llama_aid_tagging = LlamaAidTagging(crisis_type, model)
    result = llama_aid_tagging.text_to_mapping(texts)
    return AidTags(result=result)


@router.get("/get-segmentation")
async def get_segmentation(request: Request, event_id: str = "valencia-flood",
                       response_model=Segmentation) -> Segmentation:
    texts = [x for x in db if x["id"] == event_id][0]
    points = texts["points"]
    point_labels = texts["point_labels"]
    opened_image_url = texts["seg_image"]
    sam_segmentation = SAM_segment()
    result = sam_segmentation.custom_inference(points, point_labels, opened_image_url) 

    plt.imshow(result[0])
    buf = BytesIO()
    plt.savefig(buf, bbox_inches='tight', pad_inches=0)
    img_byte_arr = buf.getvalue()

    return Response(
        content=img_byte_arr, 
        media_type="image/png"
    )
