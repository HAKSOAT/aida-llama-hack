from typing import Dict, List, Optional

import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, MllamaProcessor

from backend.apis.config import Config
from backend.apis.inference import Inference


class ModelManager:
    def __init__(self, config: Config):
        self.config = config
        self.model = MllamaForConditionalGeneration.from_pretrained(
            config.MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16
        )
        self.model.tie_weights()
        self.processor = MllamaProcessor.from_pretrained(config.MODEL_NAME)

    def __del__(self) -> None:
        # Cleanup
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "processor"):
            del self.processor
        torch.cuda.empty_cache()


class LlamaClassification(Inference):
    def __init__(self, model: ModelManager):
        self.model = model

    def image_to_text(self, input: List[Image.Image]) -> List[str]:
        # Horizontally combine images with a white space between them
        combined_image = Image.new(
            "RGB",
            (sum(image.width for image in input), max(image.height for image in input)),
        )
        x_offset = 0
        for image in input:
            combined_image.paste(image, (x_offset, 0))
            x_offset += image.width + 10

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": (
                            "There has been a natural disaster between the two satellite images, "
                            "which are before (left) and after (right) images. What is the disaster? "
                            "Output only one word, for example - flood, hurricane, drought, wildfire, etc."
                        ),
                    },
                ],
            },
        ]

        prompt = self.model.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        inputs = self.model.processor(combined_image, prompt, return_tensors="pt").to(
            self.model.model.device
        )
        output = self.model.model.generate(
            **inputs,
            temperature=self.model.config.TEMPERATURE,
            top_p=self.model.config.TOP_P,
            max_new_tokens=self.model.config.MAX_TOKENS,
        )
        response = self.model.processor.decode(output[0], skip_special_tokens=True)
        text = response.split("assistant\n\n")[-1].strip().strip(".")
        return [text]

    def text_to_image(self, input: List[str]) -> List[Image.Image]:
        raise NotImplementedError("Not implemented for LlamaClassification")

    def text_to_text(self, input: List[str]) -> List[str]:
        raise NotImplementedError("Not implemented for LlamaClassification")


class LlamaSummarization(Inference):
    def __init__(self, model: ModelManager):
        self.model = model

    def image_to_text(self, input: List[Image.Image]) -> List[str]:
        raise NotImplementedError("Not implemented for LlamaSummarization")

    def text_to_image(self, input: List[str]) -> List[Image.Image]:
        raise NotImplementedError("Not implemented for LlamaSummarization")

    def text_to_text(self, input: List[str]) -> List[str]:
        raise NotImplementedError("Not implemented for LlamaSummarization")


class LlamaCaption(Inference):
    def __init__(self, disaster: str, model: ModelManager):
        self.model = model
        self.disaster = disaster

    def image_to_text(self, input: List[Image.Image]) -> List[str]:
        if not input:
            return []

        # Create all conversations at once
        conversations = []
        for i in range(len(input)):
            user_prompt = (
                f"Return output only as a JSON string with the keys Title and Description. "
                f"This is an image of a {self.disaster.lower()} disaster. Give me a description of the image, "
                f"as one of the keys in the json output. The description should be a meaningful sentence string "
                f"that someone can search upon. Then, give a 2-3 word title describing the image, as the second "
                f"key in the json output. Only give the JSON as a string, no backticks, nothing else."
                f'For example: {{"Title": "Flooded houses", "Description": "A flood has occurred in the area."}}'
            )
            conversations.append(
                [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": "You are an intelligent agent who is able to understand satellite images and detect accurately whether there is a natural disaster happening.",
                            },
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": user_prompt},
                        ],
                    },
                ]
            )

        # Process all prompts at once
        prompts = [
            self.model.processor.apply_chat_template(
                conv, add_generation_prompt=True, tokenize=False
            )
            for conv in conversations
        ]

        # Batch process all inputs
        inputs = self.model.processor(
            input, prompts, return_tensors="pt", padding=True
        ).to(self.model.model.device)
        outputs = self.model.model.generate(
            **inputs,
            temperature=self.model.config.TEMPERATURE,
            top_p=self.model.config.TOP_P,
            max_new_tokens=self.model.config.MAX_TOKENS,
        )

        # Decode all outputs
        responses = self.model.processor.batch_decode(outputs, skip_special_tokens=True)
        texts = [response.split("assistant\n\n")[-1].strip() for response in responses]
        return texts

    def text_to_image(self, input: List[str]) -> List[Image.Image]:
        raise NotImplementedError("Not implemented for LlamaDescription")

    def text_to_text(self, input: List[str]) -> List[str]:
        raise NotImplementedError("Not implemented for LlamaDescription")


class LlamaAidTagging(Inference):
    def __init__(
        self,
        disaster: str,
        model: ModelManager,
        aid_resources: Optional[List[str]] = None,
    ):
        self.manager = model
        self.disaster = disaster
        if aid_resources:
            self.aid_resources = aid_resources
        else:
            self.aid_resources = [
                "Food",
                "Medication",
                "Blankets",
                "Water",
                "Shelter",
                "Rescue transport",
            ]

    def image_to_text(self, input: List[Image.Image]) -> List[str]:
        raise NotImplementedError("Not implemented for LlamaAidTagging")

    def text_to_image(self, input: List[str]) -> List[Image.Image]:
        raise NotImplementedError("Not implemented for LlamaAidTagging")

    def text_to_text(self, input: List[str]) -> List[str]:
        conversation = []
        for commentary in input:
            formated_aid_resources = "', '".join(self.aid_resources)
            user_prompt = (
                f"{commentary}\n "
                f"There is a {self.disaster} disaster happening. "
                f"This is a message from someone who is a victim in a distressed situation. "
                f"Can you extract out the aid resources they require into a python list of strings? "
                f"Output only the python list of strings, no extracontext, no backticks, nothing else. "
                f"The extracted resources must be one of the following items: ['{formated_aid_resources}']."
            )
            system_prompt = (
                "You are an intelligent agent who is able to understand satellite, drone and ground images "
                "and detect accurately whether there is a natural disaster happening."
            )
            conversation.append(
                [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": system_prompt},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                        ],
                    },
                ]
            )

        prompts = [
            self.manager.processor.apply_chat_template(
                conv, add_generation_prompt=True, tokenize=False
            )
            for conv in conversation
        ]

        inputs = self.manager.processor(
            None, prompts, return_tensors="pt", padding=True
        ).to(self.manager.model.device)
        outputs = self.manager.model.generate(
            **inputs,
            temperature=self.manager.config.TEMPERATURE,
            top_p=self.manager.config.TOP_P,
            max_new_tokens=self.manager.config.MAX_TOKENS,
        )
        responses = self.manager.processor.batch_decode(
            outputs, skip_special_tokens=True
        )
        batch_texts = [
            response.split("assistant\n\n")[-1].strip() for response in responses
        ]
        return batch_texts

    def text_to_mapping(self, input: List[str]) -> Dict[str, int]:
        batch_texts = self.text_to_text(input)
        aid_frequencies = {aid.lower(): 0 for aid in self.aid_resources}
        for aid in self.aid_resources:
            for texts in batch_texts:
                aid = aid.lower()
                aid_frequencies[aid] += texts.lower().count(aid)

        return aid_frequencies


class LlamaRealtimeDescription(Inference):
    def __init__(self, disaster: str, model: ModelManager):
        self.model = model
        self.disaster = disaster

    def image_to_text(self, input: List[Image.Image]) -> List[str]:
        raise NotImplementedError("Not implemented for LlamaRealtimeDescription")

    def text_to_image(self, input: List[str]) -> List[Image.Image]:
        raise NotImplementedError("Not implemented for LlamaRealtimeDescription")

    def text_to_text(self, input: List[str]) -> List[str]:
        raise NotImplementedError("Not implemented for LlamaRealtimeDescription")

    def custom_inference(
        self, commentary: List[str], img_captions: List[str]
    ) -> List[str]:
        user_prompt = (
            f"Below is a time-ordered list of commentaries by distressed citizens and image descriptions"
            f"showing damage done by a {self.disaster} disaster. Create a 1 sentence summary which describes the latest and "
            f"most critical information for an aid worker to know what jobs they should prioritise (and where if necessary). "
            f"Commentaries: {commentary}. Image Descriptions: {img_captions}. Be specific about the summary of the situation. "
            f"Your response should focus on the key information, and not intro or outro fluff."
        )

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]

        prompt = self.model.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        inputs = self.model.processor(None, prompt, return_tensors="pt").to(
            self.model.model.device
        )
        output = self.model.model.generate(
            **inputs,
            temperature=self.model.config.TEMPERATURE,
            top_p=self.model.config.TOP_P,
            max_new_tokens=self.model.config.MAX_TOKENS,
        )
        response = self.model.processor.decode(output[0], skip_special_tokens=True)
        text = response.split("assistant\n\n")[-1].strip()
        text = " ".join(text.split())
        return [text]
