#!/usr/bin/env python3
"""
Text Analyzer Component - AI-OS SDK Example
A complete implementation of a text analysis tool component.
"""

from typing import Dict, Any, List, Optional
import re
import time
from collections import Counter
from vertixia_sdk import AIToolComponent


class TextAnalyzer(AIToolComponent):
    """
    Text Analyzer Component
    
    Analyzes text for sentiment, readability, and key metrics.
    This component demonstrates:
    - Input/output validation
    - Parameter handling
    - Error management
    - Performance tracking
    """
    
    def _tool_execute(self, text: str, analysis_type: str = "comprehensive", 
                     include_metrics: bool = True, language: str = "auto", 
                     **kwargs) -> Dict[str, Any]:
        """
        Execute text analysis
        
        Args:
            text: Text content to analyze
            analysis_type: Type of analysis ("basic", "sentiment", "readability", "comprehensive")
            include_metrics: Include detailed metrics in output
            language: Text language for analysis
            
        Returns:
            Dictionary containing analysis results
        """
        start_time = time.time()
        
        try:
            # Input validation
            if not text or not text.strip():
                raise ValueError("Text cannot be empty")
            
            if len(text) > 10000:
                raise ValueError("Text exceeds maximum length of 10,000 characters")
            
            # Clean text
            cleaned_text = text.strip()
            
            # Initialize results
            results = {
                "processing_time": 0.0
            }
            
            # Perform analysis based on type
            if analysis_type in ["basic", "comprehensive"]:
                results.update(self._basic_analysis(cleaned_text))
            
            if analysis_type in ["sentiment", "comprehensive"]:
                results.update(self._sentiment_analysis(cleaned_text))
            
            if analysis_type in ["readability", "comprehensive"]:
                results.update(self._readability_analysis(cleaned_text))
            
            if include_metrics:
                results.update(self._extract_metrics(cleaned_text))
            
            # Calculate processing time
            results["processing_time"] = round(time.time() - start_time, 3)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Text analysis failed: {str(e)}")
            raise
    
    def _basic_analysis(self, text: str) -> Dict[str, Any]:
        """Perform basic text analysis"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "character_count": len(text),
            "character_count_no_spaces": len(text.replace(" ", "")),
            "average_words_per_sentence": round(len(words) / max(len(sentences), 1), 2)
        }
    
    def _sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Perform sentiment analysis (simplified implementation)"""
        # Simplified sentiment analysis using keyword matching
        positive_words = {
            "good", "great", "excellent", "amazing", "wonderful", "fantastic",
            "love", "like", "enjoy", "happy", "pleased", "satisfied", "awesome"
        }
        
        negative_words = {
            "bad", "terrible", "awful", "horrible", "hate", "dislike",
            "sad", "angry", "disappointed", "frustrated", "annoying", "worst"
        }
        
        words = set(text.lower().split())
        positive_count = len(words.intersection(positive_words))
        negative_count = len(words.intersection(negative_words))
        
        if positive_count > negative_count:
            sentiment = "positive"
            confidence = min(0.9, 0.5 + (positive_count - negative_count) * 0.1)
        elif negative_count > positive_count:
            sentiment = "negative"
            confidence = min(0.9, 0.5 + (negative_count - positive_count) * 0.1)
        else:
            sentiment = "neutral"
            confidence = 0.5
        
        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 2),
            "positive_indicators": positive_count,
            "negative_indicators": negative_count
        }
    
    def _readability_analysis(self, text: str) -> Dict[str, Any]:
        """Perform readability analysis"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Count syllables (simplified)
        syllable_count = 0
        for word in words:
            word = word.lower().strip('.,!?;:"()[]{}')
            syllable_count += max(1, len(re.findall(r'[aeiouAEIOU]', word)))
        
        # Flesch Reading Ease Score (simplified)
        avg_sentence_length = len(words) / max(len(sentences), 1)
        avg_syllables_per_word = syllable_count / max(len(words), 1)
        
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        flesch_score = max(0, min(100, flesch_score))
        
        # Grade level interpretation
        if flesch_score >= 90:
            grade_level = "5th grade"
        elif flesch_score >= 80:
            grade_level = "6th grade"
        elif flesch_score >= 70:
            grade_level = "7th grade"
        elif flesch_score >= 60:
            grade_level = "8th-9th grade"
        elif flesch_score >= 50:
            grade_level = "10th-12th grade"
        elif flesch_score >= 30:
            grade_level = "College level"
        else:
            grade_level = "Graduate level"
        
        return {
            "readability_score": round(flesch_score, 1),
            "grade_level": grade_level,
            "avg_sentence_length": round(avg_sentence_length, 1),
            "avg_syllables_per_word": round(avg_syllables_per_word, 1)
        }
    
    def _extract_metrics(self, text: str) -> Dict[str, Any]:
        """Extract additional text metrics"""
        words = text.split()
        
        # Word frequency
        word_freq = Counter(word.lower().strip('.,!?;:"()[]{}') for word in words)
        most_common = word_freq.most_common(5)
        
        # Extract key phrases (simple noun phrases)
        key_phrases = self._extract_key_phrases(text)
        
        return {
            "unique_words": len(word_freq),
            "most_common_words": [{"word": word, "count": count} for word, count in most_common],
            "key_phrases": key_phrases,
            "lexical_diversity": round(len(word_freq) / max(len(words), 1), 3)
        }
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text (simplified implementation)"""
        # Simple extraction of capitalized phrases and noun phrases
        phrases = []
        
        # Find capitalized phrases
        capitalized_phrases = re.findall(r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b', text)
        phrases.extend(capitalized_phrases[:3])
        
        # Find potential noun phrases (simplified)
        words = text.split()
        for i in range(len(words) - 1):
            if len(words[i]) > 3 and len(words[i+1]) > 3:
                phrase = f"{words[i]} {words[i+1]}"
                if phrase.lower() not in [p.lower() for p in phrases]:
                    phrases.append(phrase)
                    if len(phrases) >= 5:
                        break
        
        return phrases[:5]


# Example usage and testing
if __name__ == "__main__":
    # Create component instance
    analyzer = TextAnalyzer()
    
    # Example text
    sample_text = """
    AI-OS is an incredible operating system that provides amazing capabilities
    for automation and intelligent task management. The system is designed to
    learn from user interactions and improve over time. It offers excellent
    performance and wonderful user experience.
    """
    
    # Test different analysis types
    print("=== Comprehensive Analysis ===")
    result = analyzer.execute(
        text=sample_text.strip(),
        analysis_type="comprehensive",
        include_metrics=True
    )
    
    for key, value in result.items():
        print(f"{key}: {value}")
    
    print("\n=== Basic Analysis ===")
    basic_result = analyzer.execute(
        text=sample_text.strip(),
        analysis_type="basic",
        include_metrics=False
    )
    
    for key, value in basic_result.items():
        print(f"{key}: {value}")
    
    print("\n=== Sentiment Analysis ===")
    sentiment_result = analyzer.execute(
        text=sample_text.strip(),
        analysis_type="sentiment"
    )
    
    for key, value in sentiment_result.items():
        print(f"{key}: {value}")