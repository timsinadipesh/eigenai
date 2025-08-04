"""
Piper TTS integration for text-to-speech with robust language detection
"""

import os
import re
import hashlib
import subprocess
import logging
from pathlib import Path
from typing import Optional
from langdetect import detect, LangDetectException, DetectorFactory

# Set seed for consistent language detection results
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)


class PiperTTSEngine:
    """Piper TTS integration for text-to-speech with robust language detection"""
    
    def __init__(self):
        # Find the voices directory relative to this file
        self.voices_dir = Path(__file__).parent / "voices"
        self.cache_dir = Path.home() / ".cache" / "piper_audio"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Dynamically find all .onnx voice models in the voices directory
        self.voices = {}
        if self.voices_dir.exists():
            for onnx_file in self.voices_dir.glob("*.onnx"):
                # Example: en_US-lessac-high.onnx → "en_US" and "en"
                lang_full = onnx_file.name.split("-")[0]  # "en_US"
                lang_short = lang_full.split("_")[0]      # "en"
                self.voices[lang_full] = onnx_file
                self.voices[lang_short] = onnx_file
                logger.info(f"Found voice model: {lang_short} -> {onnx_file.name}")

        # Check if piper is available
        self.piper_available = self._check_piper_installation()
        
        # Pre-compile regex for text cleaning
        self._text_cleaner = re.compile(r'[^\w\s\.\,\!\?\-\']')
        
        # Language detection cache for performance
        self._detection_cache = {}
        self._cache_max_size = 100
        
        # Language-specific keywords for short text detection
        self.lang_keywords = {
            'fr': {
                'articles': ['le', 'la', 'les', 'un', 'une', 'des', 'du'],
                'common': ['et', 'est', 'avec', 'pour', 'dans', 'sur', 'que', 'qui', 'mais', 'vous', 'nous', 'ils'],
                'pronouns': ['je', 'tu', 'il', 'elle', 'on'],
            },
            'de': {
                'articles': ['der', 'die', 'das', 'ein', 'eine', 'den', 'dem'],
                'common': ['und', 'ist', 'mit', 'für', 'auf', 'von', 'zu', 'nicht', 'ich', 'sie', 'wir'],
                'pronouns': ['ich', 'du', 'er', 'sie', 'es', 'wir', 'ihr'],
            }
        }
        
    def _check_piper_installation(self) -> bool:
        """Check if piper is installed and available"""
        try:
            result = subprocess.run(["piper", "--help"], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("Piper TTS not found. TTS functionality disabled.")
            return False
    
    def _detect_language_short_text(self, text: str) -> Optional[str]:
        """Detect language for short texts using keyword analysis"""
        text_lower = text.lower()
        words = set(text_lower.split())
        
        # Score each language based on keyword matches
        scores = {}
        
        for lang, keywords in self.lang_keywords.items():
            score = 0
            # Check articles (strong indicators)
            for article in keywords['articles']:
                if article in words:
                    score += 3
            # Check common words
            for common in keywords['common']:
                if common in words:
                    score += 2
            # Check pronouns
            for pronoun in keywords['pronouns']:
                if pronoun in words:
                    score += 2
            
            if score > 0:
                scores[lang] = score
        
        # Return language with highest score if confident enough
        if scores:
            best_lang = max(scores, key=scores.get)
            if scores[best_lang] >= 4:  # Confidence threshold
                return best_lang
        
        return None
    
    def _detect_language(self, text: str) -> str:
        """
        Robust language detection with fallback strategies.
        Returns 'en', 'fr', or 'de' based on available voices.
        """
        try:
            # Check cache first
            text_sample = text[:200]  # Use first 200 chars for cache key
            cache_key = hash(text_sample)
            if cache_key in self._detection_cache:
                return self._detection_cache[cache_key]
            
            # Clean text for detection
            clean_text = self._text_cleaner.sub(' ', text).strip()
            
            # Strategy 1: Short text detection using keywords
            if len(clean_text) < 30:
                detected = self._detect_language_short_text(clean_text)
                if detected and detected in self.voices:
                    logger.info(f"Short text detection: {detected}")
                    self._detection_cache[cache_key] = detected
                    return detected
                # For very short texts without clear indicators, default to English
                if len(clean_text) < 15:
                    logger.info("Very short text, defaulting to English")
                    return 'en'
            
            # Strategy 2: Use langdetect for longer texts
            try:
                detected = detect(clean_text)
                logger.info(f"Langdetect result: {detected}")
                
                # Map detected language to available voices
                if detected in ['en', 'fr', 'de']:
                    if detected in self.voices and self.voices[detected].exists():
                        # Cache the result
                        if len(self._detection_cache) < self._cache_max_size:
                            self._detection_cache[cache_key] = detected
                        return detected
                    else:
                        logger.warning(f"Voice for {detected} not available")
                
                # Handle other language codes that might map to our supported languages
                lang_mapping = {
                    'english': 'en',
                    'french': 'fr', 
                    'german': 'de',
                    'deutsch': 'de',
                    'francais': 'fr',
                    'français': 'fr'
                }
                
                if detected.lower() in lang_mapping:
                    mapped_lang = lang_mapping[detected.lower()]
                    if mapped_lang in self.voices:
                        self._detection_cache[cache_key] = mapped_lang
                        return mapped_lang
                
            except LangDetectException as e:
                logger.warning(f"Langdetect failed: {e}")
            
            # Strategy 3: Fallback keyword detection for any length
            if len(clean_text) >= 30:
                detected = self._detect_language_short_text(clean_text)
                if detected and detected in self.voices:
                    logger.info(f"Fallback keyword detection: {detected}")
                    self._detection_cache[cache_key] = detected
                    return detected
            
            # Default to English
            logger.info("All detection strategies failed, defaulting to English")
            return 'en'
            
        except Exception as e:
            logger.error(f"Language detection error: {e}")
            return 'en'
    
    def _get_cache_path(self, text: str, language: str) -> Path:
        """Generate cache file path based on text hash and language"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return self.cache_dir / f"{text_hash}_{language}.wav"
    
    async def synthesize(self, text: str, language: Optional[str] = None) -> Optional[Path]:
        """
        Synthesize text to speech using Piper.
        
        Args:
            text: The text to synthesize
            language: Optional language code ('en', 'fr', 'de'). If None, auto-detects.
        
        Returns:
            Path to the synthesized audio file, or None if synthesis failed
        """
        
        if not self.piper_available:
            logger.error("Piper TTS is not available")
            return None
        
        # Handle language parameter
        if language is not None:
            # Validate provided language
            if language not in self.voices:
                logger.warning(f"Requested language '{language}' not available, auto-detecting instead")
                language = self._detect_language(text)
            elif not self.voices[language].exists():
                logger.warning(f"Voice model for '{language}' not found, auto-detecting instead")
                language = self._detect_language(text)
            else:
                logger.info(f"Using requested language: {language}")
        else:
            # Auto-detect language
            language = self._detect_language(text)
            logger.info(f"Auto-detected language: {language}")
        
        # Final validation
        if language not in self.voices or not self.voices[language].exists():
            logger.warning(f"Voice model for {language} not found, falling back to English")
            language = "en"
            if language not in self.voices or not self.voices[language].exists():
                logger.error("No voice models available")
                return None
        
        # Check cache first
        cache_path = self._get_cache_path(text, language)
        if cache_path.exists():
            logger.info(f"Using cached audio: {cache_path}")
            return cache_path
        
        try:
            # Clean text for TTS
            clean_text = self._clean_text_for_tts(text)
            
            # Run Piper TTS
            cmd = [
                "piper",
                "--model", str(self.voices[language]),
                "--output_file", str(cache_path)
            ]
            
            logger.info(f"Running Piper TTS with model: {self.voices[language].name}")
            
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(input=clean_text, timeout=30)
            
            if process.returncode == 0 and cache_path.exists():
                file_size = cache_path.stat().st_size
                logger.info(f"TTS synthesis complete: {cache_path} ({file_size} bytes)")
                return cache_path
            else:
                logger.error(f"Piper TTS failed with return code {process.returncode}")
                if stderr:
                    logger.error(f"Piper stderr: {stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            process.kill()
            logger.error("Piper TTS timed out after 30 seconds")
            return None
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            return None
    
    def _clean_text_for_tts(self, text: str) -> str:
        """Clean text for better TTS synthesis - no length limits"""
        # Remove markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic  
        text = re.sub(r'#{1,6}\s', '', text)          # Headers
        
        # Remove code blocks
        text = re.sub(r'```[^`]*```', ' ', text)
        
        # Remove multiple spaces and newlines
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Replace smart quotes and other special characters
        replacements = {
            '"': '"', '"': '"', ''': "'", ''': "'",
            '–': '-', '—': '-', '…': '...'
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def clear_cache(self, language: Optional[str] = None):
        """Clear TTS cache, optionally for a specific language"""
        try:
            if language:
                # Clear only files for specific language
                pattern = f"*_{language}.wav"
                files = list(self.cache_dir.glob(pattern))
                for file in files:
                    file.unlink()
                logger.info(f"Cleared {len(files)} cached files for language: {language}")
            else:
                # Clear all cache
                files = list(self.cache_dir.glob("*.wav"))
                for file in files:
                    file.unlink()
                logger.info(f"Cleared {len(files)} cached files")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")