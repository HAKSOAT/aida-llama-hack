from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, Request
from PIL import Image

from backend.apis import mock_images
from backend.apis.llama import (
    LlamaAidTagging,
    LlamaCaption,
    LlamaClassification,
    LlamaRealtimeDescription,
)
from backend.schemas.response_schema import CrisisType, Caption, AidTags, Commentary, RealtimeDescription
router = APIRouter()
logger = logging.getLogger(__name__)


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
    response_model=CrisisType
) -> CrisisType:
    image_directory = iter(
        (Path(mock_images.__path__[0]) / "crisis_type").glob("*.webp")
    )
    images = [Image.open(str(file)) for file in image_directory]
    model = request.app.state.llama_model
    llama_classification = LlamaClassification(model)
    result = llama_classification.image_to_text(images)
    text = result[0]
    return CrisisType(result=text)


@router.get("/get-caption")
async def get_caption(request: Request, 
                      response_model=Caption) -> Caption:
    target_extensions = ["*.jpg", "*.png"]
    images = [
        Image.open(str(file))
        for ext in target_extensions
        for file in (Path(mock_images.__path__[0]) / "caption").glob(ext)
    ]
    model = request.app.state.llama_model
    llama_caption = LlamaCaption("flood", model)
    result = llama_caption.image_to_text(images)
    obj = {k.lower(): v for k, v in json.loads(result[0]).items()}
    return Caption(result=obj)


@router.get("/get-segmentation")
async def get_segmentation(request: Request) -> None:
    pass


@router.get("/{event_id}/description")
async def get_description(request: Request, event_id: str, 
                          response_model=RealtimeDescription) -> RealtimeDescription:
    model = request.app.state.llama_model
    llama_realtime_description = LlamaRealtimeDescription("flood", model)
    commentary = [
        "I'm alone, and the power is out. I don't have enough medication or food to last much longer. Please, can someone come and check on us here?",
        "The bridge near us has collapsed, and we're stranded. My husband is injured, and needs first-aid. We need food, clean water, and blankets for the kids. Can someone help us get supplies?",
        "The water flow has subsided in the west part of the city now. The wind has also slowed down.",
    ]
    img_captions = [""]
    result = llama_realtime_description.custom_inference(commentary, img_captions)
    return RealtimeDescription(result=result[0])


@router.get("/get-aid-tags")
async def get_aid_tags(request: Request, 
                       response_model=AidTags) -> AidTags:
    texts = [
        "I'm alone, and the power is out. I don't have enough medication or food to last much longer. Please, can someone come and check on us here?",
        "The bridge near us has collapsed, and we're stranded. My husband is injured, and needs first-aid. We need food, clean water, and blankets for the kids. Can someone help us get supplies?",
    ]
    model = request.app.state.llama_model
    llama_aid_tagging = LlamaAidTagging("flood", model)
    result = llama_aid_tagging.text_to_mapping(texts)
    return AidTags(result=result)
