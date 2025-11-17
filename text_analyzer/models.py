from pydantic import BaseModel
from typing import List, Optional
from enum import Enum

class ToneType(str, Enum):
    FORMAL = "formal"
    INFORMAL = "informal"
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    COMPLAINT = "complaint"
    PERSUASIVE = "persuasive"
    ACADEMIC = "academic"

class AnalysisRequest(BaseModel):
    text: str
    target_tone: Optional[ToneType] = None

class AnalysisResponse(BaseModel):
    moral: str
    keywords: List[str]
    transformed_text: Optional[str] = None
    original_tone: str
    target_tone: Optional[str] = None
    confidence: float

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None