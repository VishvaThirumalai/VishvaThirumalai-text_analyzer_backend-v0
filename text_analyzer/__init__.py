"""
Text Analyzer API Package

A FastAPI application for AI-powered text analysis including:
- Moral extraction from paragraphs
- Keyword and phrase identification  
- Tone analysis and transformation
- Multiple tone styles (formal, informal, friendly, etc.)
"""

__version__ = "1.0.0"
__author__ = "Text Analyzer Team"

from .main import app
from .models import AnalysisRequest, AnalysisResponse, ToneType
from .services import TextAnalysisService

__all__ = [
    "app",
    "AnalysisRequest", 
    "AnalysisResponse",
    "ToneType",
    "TextAnalysisService"
]