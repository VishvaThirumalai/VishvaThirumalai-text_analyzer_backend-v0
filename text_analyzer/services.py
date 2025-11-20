import os
import logging
from typing import List, Dict, Any
from rake_nltk import Rake
from nltk.tokenize import sent_tokenize
import nltk

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------
#  TEXT ANALYSIS SERVICE CLASS
# -----------------------------

class TextAnalysisService:
    def __init__(self):
        """
        Initialize OpenAI, NLTK and RAKE keyword extractor.
        Compatible with OpenAI Python SDK >= 1.0
        """

        # Load OpenAI key
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            logger.warning("⚠ OPENAI_API_KEY not found. Running in DEMO MODE.")
            self.demo_mode = True
            self.client = None
        else:
            from openai import OpenAI
            # Initialize with just the API key - no extra parameters
            self.client = OpenAI(api_key=api_key)
            self.demo_mode = False
            logger.info("✅ OpenAI client initialized")

        # Initialize NLTK
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords")

        # Initialize RAKE
        try:
            self.rake = Rake()
        except Exception as e:
            logger.warning(f"RAKE initialization failed: {e}")
            self.rake = None

    # ----------------------------------------------------------------------
    # 1. SUMMARY / MORAL EXTRACTOR
    # ----------------------------------------------------------------------

    def extract_moral(self, text: str) -> str:
        """
        Extract a short moral or main lesson from the text.
        """
        if self.demo_mode:
            return "Demo moral: Always be kind and responsible."

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # Using gpt-3.5-turbo for cost efficiency
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful text analyzer. Extract the main moral or lesson from the given text in 1-2 sentences."
                    },
                    {
                        "role": "user", 
                        "content": f"Text to analyze: {text}"
                    }
                ],
                max_tokens=100,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating moral: {e}")
            return "Error generating moral."

    # ----------------------------------------------------------------------
    # 2. KEYWORD EXTRACTOR
    # ----------------------------------------------------------------------

    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract important keywords from the text using RAKE.
        """
        if not self.rake:
            return ["keyword extraction unavailable"]

        try:
            self.rake.extract_keywords_from_text(text)
            keywords = self.rake.get_ranked_phrases()[:8]
            return keywords
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            return []

    # ----------------------------------------------------------------------
    # 3. TONALITY MODIFY FUNCTION
    # ----------------------------------------------------------------------

    def change_tone(self, text: str, tone: str) -> str:
        """
        Convert text tonality to: formal, informal, complaint, friendly, etc.
        """
        if self.demo_mode:
            return f"[Demo] Converted to {tone} tone: {text[:100]}..."

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a tone converter. Rewrite the following text in a {tone} tone while preserving the original meaning."
                    },
                    {
                        "role": "user", 
                        "content": text
                    }
                ],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Tone conversion failed: {e}")
            return "Error converting tone."

    # ----------------------------------------------------------------------
    # 4. FULL PROCESSOR – COMBINES EVERYTHING
    # ----------------------------------------------------------------------

    def analyze_text(self, text: str, tone: str = None) -> Dict[str, Any]:
        """
        Complete pipeline:
        - Moral extraction
        - Keyword extraction
        - Tone transformation (optional)
        """
        result = {}

        # Validate input
        if not text or len(text.strip()) == 0:
            result["error"] = "Text cannot be empty"
            return result

        result["moral"] = self.extract_moral(text)
        result["keywords"] = self.extract_keywords(text)

        if tone:
            result["converted_text"] = self.change_tone(text, tone)
        else:
            result["converted_text"] = None

        return result


# Global shared service instance
text_service = TextAnalysisService()
