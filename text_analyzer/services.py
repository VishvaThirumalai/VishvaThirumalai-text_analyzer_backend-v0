import os
import asyncio
import logging
from typing import Optional
import random
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextAnalysisService:
    def __init__(self):
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not found. Using demo mode.")
            self.demo_mode = True
            self.client = None
        else:
            self.demo_mode = False
            self.client = OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized successfully")

        # Initialize NLTK + RAKE
        try:
            import nltk
            from rake_nltk import Rake

            try: nltk.data.find("tokenizers/punkt")
            except LookupError: nltk.download("punkt")

            try: nltk.data.find("corpora/stopwords")
            except LookupError: nltk.download("stopwords")

            self.rake = Rake()
        except Exception as e:
            logger.warning(f"Keyword extractor failed: {e}")
            self.rake = None

    async def analyze_text(self, text: str, target_tone: str = None) -> dict:
        keywords = await self._extract_keywords(text)
        tone = await self._analyze_sentiment(text)
        moral = await self._extract_moral(text)

        transformed_text = None
        if target_tone:
            transformed_text = await self._transform_tone(text, target_tone)

        return {
            "moral": moral,
            "keywords": keywords,
            "transformed_text": transformed_text,
            "original_tone": tone["label"],
            "target_tone": target_tone,
            "confidence": tone["score"]
        }

    async def _extract_keywords(self, text: str) -> list:
        if self.rake:
            try:
                self.rake.extract_keywords_from_text(text)
                return self.rake.get_ranked_phrases()[:10]
            except:
                pass

        words = text.lower().split()
        common = {'the','a','an','and','or','but','in','on','at','to','for','of','with','by','is','are'}
        return [w for w in words if w not in common and len(w) > 3][:10]

    async def _analyze_sentiment(self, text: str) -> dict:
        try:
            from transformers import pipeline
            sentiment = pipeline("sentiment-analysis")
            r = sentiment(text[:512])[0]
            return {
                "label": "positive" if "POS" in r["label"] else "negative",
                "score": float(r["score"])
            }
        except:
            return {"label": "neutral", "score": 0.5}

    async def _extract_moral(self, text: str) -> str:
        if self.demo_mode:
            return self._demo_moral_extraction(text)

        try:
            resp = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{
                        "role": "user",
                        "content": f"Extract the moral lesson in 1–2 sentences:\n{text[:1500]}"
                    }]
                )
            )
            return resp.choices[0].message["content"].strip()

        except Exception as e:
            logger.error(f"Moral extraction failed: {e}")
            return self._demo_moral_extraction(text)

    async def _transform_tone(self, text: str, target_tone: str) -> str:
        if self.demo_mode:
            return self._demo_tone_transformation(text, target_tone)

        prompts = {
            "formal": "Rewrite formally:",
            "informal": "Rewrite casually:",
            "friendly": "Rewrite in friendly tone:",
            "complaint": "Rewrite as a complaint:",
            "professional": "Rewrite in business tone:",
            "persuasive": "Rewrite persuasively:",
            "academic": "Rewrite academically:",
            "casual": "Rewrite casually:"
        }

        instruction = prompts.get(target_tone, "Rewrite this:")

        try:
            resp = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{
                        "role": "user",
                        "content": f"{instruction}\n\n{text[:1500]}"
                    }]
                )
            )
            return resp.choices[0].message["content"].strip()

        except Exception as e:
            logger.error(f"Tone transformation failed: {e}")
            return self._demo_tone_transformation(text, target_tone)

    def _demo_moral_extraction(self, text: str) -> str:
        themes = [
            "perseverance", "honesty", "friendship",
            "kindness", "growth", "courage"
        ]
        return f"This story teaches a lesson about {random.choice(themes)}."

    def _demo_tone_transformation(self, text: str, tone: str) -> str:
        return f"[{tone.upper()} VERSION] {text[:200]}..."

# Global instance
text_service = TextAnalysisService()
