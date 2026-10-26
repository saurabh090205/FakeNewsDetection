from typing import Literal, Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    headline: str = Field(min_length=1, max_length=1000)
    article: str = Field(min_length=1, max_length=100000)
    model: str = Field(min_length=1)


class Prediction(BaseModel):
    prediction: Literal["REAL", "FAKE", "UNCERTAIN"]
    confidence: float
    real_probability: float
    fake_probability: float
    model: str
    explanation_note: str
    predicted_label_display: Optional[str] = None
    article_words: Optional[int] = None
    article_chars: Optional[int] = None
    cleaned_text_preview: Optional[str] = None
    timestamp: Optional[str] = None


class WordWeight(BaseModel):
    word: str
    weight: float
    direction: str = "supports"
    abs_weight: float = 0.0


class ExplanationResponse(BaseModel):
    model: str
    prediction: str
    prediction_label: str = ""
    confidence: float = 0.0
    words: list[WordWeight] = []
    positive_words: list[WordWeight] = []
    negative_words: list[WordWeight] = []
    plain_english_summary: str = ""
    note: str
    methodology: str = ""
    cached: bool = False
    execution_time_ms: float = 0.0


class ErrorResponse(BaseModel):
    error: str
    code: str

