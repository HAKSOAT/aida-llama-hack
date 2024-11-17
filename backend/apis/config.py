from pydantic import BaseModel


class Config(BaseModel):
    MODEL_NAME: str
    TEMPERATURE: float
    TOP_P: float
    MAX_TOKENS: int = 512


class MllamaConfig(Config):
    MODEL_NAME: str = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    TEMPERATURE: float = 0.1
    TOP_P: float = 0.1
