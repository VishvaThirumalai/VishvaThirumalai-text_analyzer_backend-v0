import os
import asyncio
import logging
from typing import Optional
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextAnalysisService:
    def __init__(self):
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or api_key == "your_openai_api_key_here":
            logger.warning("OPENAI_API_KEY not found. Using enhanced demo mode.")
            self.demo_mode = True
        else:
            self.demo_mode = False
            try:
                import openai
                self.openai_client = openai.OpenAI(api_key=api_key)
                logger.info("OpenAI client initialized successfully")
            except ImportError:
                logger.warning("OpenAI package not available. Using demo mode.")
                self.demo_mode = True
        
        # Initialize NLTK for keyword extraction
        try:
            import nltk
            from rake_nltk import Rake
            
            # Download required NLTK data
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
            
            try:
                nltk.data.find('corpora/stopwords') 
            except LookupError:
                nltk.download('stopwords')
                
            self.rake = Rake()
            logger.info("Keyword extractor initialized successfully")
        except ImportError as e:
            logger.warning(f"Could not initialize keyword extractor: {e}")
            self.rake = None

    async def analyze_text(self, text: str, target_tone: str = None) -> dict:
        """Analyze text and extract moral, keywords, and optionally transform tone"""
        
        # Extract keywords
        keywords = await self._extract_keywords(text)
        
        # Analyze sentiment/tone
        tone_analysis = await self._analyze_sentiment(text)
        
        # Extract moral
        moral = await self._extract_moral(text)
        
        # Transform tone if requested
        transformed_text = None
        if target_tone:
            transformed_text = await self._transform_tone(text, target_tone)
        
        return {
            "moral": moral,
            "keywords": keywords,
            "transformed_text": transformed_text,
            "original_tone": tone_analysis['label'],
            "target_tone": target_tone,
            "confidence": tone_analysis['score']
        }
    
    async def _extract_keywords(self, text: str) -> list:
        """Extract keywords using RAKE or fallback"""
        if self.rake:
            try:
                self.rake.extract_keywords_from_text(text)
                keywords = self.rake.get_ranked_phrases()[:10]
                return keywords if keywords else ["meaningful", "content", "analysis"]
            except Exception as e:
                logger.error(f"Keyword extraction failed: {e}")
        
        # Fallback keyword extraction
        words = text.lower().split()
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        keywords = [word for word in words if word not in common_words and len(word) > 3]
        return list(set(keywords))[:10] or ["important", "key", "concepts"]
    
    async def _analyze_sentiment(self, text: str) -> dict:
        """Analyze sentiment with multiple fallbacks"""
        # Try transformers first
        try:
            from transformers import pipeline
            sentiment_analyzer = pipeline("sentiment-analysis")
            result = sentiment_analyzer(text[:512])[0]
            return {
                'label': self._map_sentiment_to_tone(result['label']),
                'score': float(result['score'])
            }
        except Exception as e:
            logger.info(f"Transformers sentiment analysis not available: {e}")
        
        # Fallback to simple analysis
        return self._simple_sentiment_analysis(text)
    
    def _simple_sentiment_analysis(self, text: str) -> dict:
        """Simple rule-based sentiment analysis"""
        positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'happy', 'love', 'nice', 'beautiful', 'perfect', 'fantastic'}
        negative_words = {'bad', 'terrible', 'awful', 'horrible', 'sad', 'hate', 'angry', 'worst', 'disappointing', 'ugly'}
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return {'label': 'positive', 'score': 0.8}
        elif negative_count > positive_count:
            return {'label': 'negative', 'score': 0.8}
        else:
            return {'label': 'neutral', 'score': 0.5}
    
    async def _extract_moral(self, text: str) -> str:
        """Extract moral/lesson from text"""
        if self.demo_mode:
            return self._demo_moral_extraction(text)
        
        try:
            prompt = f"Extract the moral or main lesson from this text in 1-2 sentences: {text[:1500]}"
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100,
                    temperature=0.3
                )
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI moral extraction failed: {e}")
            return self._demo_moral_extraction(text)
    
    def _demo_moral_extraction(self, text: str) -> str:
        """Enhanced demo moral extraction"""
        if len(text) < 20:
            return "The text is too short to extract a meaningful lesson."
        
        themes = [
            "the importance of perseverance and determination in overcoming challenges",
            "the value of friendship, teamwork, and supporting one another",
            "learning from mistakes and using experiences for personal growth", 
            "the power of kindness, empathy, and understanding others",
            "the significance of honesty, integrity, and doing what's right",
            "appreciating the simple things in life and finding contentment",
            "the courage to face fears and step outside comfort zones",
            "the wisdom in being patient and thinking before acting"
        ]
        
        return f"This story teaches us about {random.choice(themes)}."
    
    async def _transform_tone(self, text: str, target_tone: str) -> str:
        """Transform text to specified tone"""
        if self.demo_mode:
            return self._demo_tone_transformation(text, target_tone)
        
        try:
            tone_instructions = {
                "formal": "Rewrite this formally and professionally:",
                "informal": "Rewrite this casually like talking to a friend:",
                "friendly": "Rewrite this in a warm, friendly way:",
                "professional": "Rewrite this in a business-appropriate tone:",
                "casual": "Rewrite this in a relaxed, everyday tone:",
                "complaint": "Rewrite this as a formal complaint:",
                "persuasive": "Rewrite this persuasively to convince someone:",
                "academic": "Rewrite this in academic language:"
            }
            
            instruction = tone_instructions.get(target_tone, "Rewrite this text:")
            prompt = f"{instruction}\n\n{text[:1500]}"
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.7
                )
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI tone transformation failed: {e}")
            return self._demo_tone_transformation(text, target_tone)
    
    def _demo_tone_transformation(self, text: str, target_tone: str) -> str:
        """Enhanced demo tone transformation"""
        prefixes = {
            "formal": "[Formal Version] ",
            "informal": "[Casual Version] ",
            "friendly": "[Friendly Version] ",
            "professional": "[Professional Version] ",
            "casual": "[Relaxed Version] ",
            "complaint": "[Formal Complaint] ",
            "persuasive": "[Persuasive Version] ",
            "academic": "[Academic Version] "
        }
        
        prefix = prefixes.get(target_tone, "[Rewritten] ")
        
        # Smart text shortening
        if len(text) > 200:
            sentences = text.split('.')
            if len(sentences) > 1:
                shortened = '.'.join(sentences[:2]) + '.'
            else:
                shortened = text[:197] + "..."
        else:
            shortened = text
            
        return prefix + shortened
    
    def _map_sentiment_to_tone(self, sentiment_label: str) -> str:
        """Map sentiment to tone"""
        return "positive" if "POSITIVE" in sentiment_label or "LABEL_1" in sentiment_label else "negative" if "NEGATIVE" in sentiment_label or "LABEL_0" in sentiment_label else "neutral"

# Global service instance
text_service = TextAnalysisService()