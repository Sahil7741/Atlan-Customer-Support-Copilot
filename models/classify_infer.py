import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import re

TOPICS = ["How-to","Product","Connector","Lineage","API/SDK","SSO","Glossary","Best practices","Sensitive data"]
SENTIMENTS = ["Frustrated","Curious","Angry","Neutral"]
PRIORITIES = ["P0","P1","P2"]

class CustomerSupportClassifier:
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.sentiment_pipeline = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=0 if self.device == "cuda" else -1
        )
        
        self.topic_pipeline = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if self.device == "cuda" else -1
        )
        
        self.priority_pipeline = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if self.device == "cuda" else -1
        )
    
    def classify(self, text: str) -> dict:
        words = text.split()
        if len(words) > 400:
            truncated_text = " ".join(words[:400])
        else:
            truncated_text = text
        
        # Sentiment Analysis
        sentiment_result = self.sentiment_pipeline(truncated_text)
        sentiment = self._map_sentiment(sentiment_result[0]['label'], sentiment_result[0]['score'])
        
        # Topic Classification using zero-shot (BART-MNLI also has token limits)
        topic_result = self.topic_pipeline(truncated_text, TOPICS)
        topic_tags = self._extract_topics(topic_result)
        
        # Enhance with keyword-based rules for better accuracy
        topic_tags = self._enhance_topics(text, topic_tags)
        
        priority_result = self.priority_pipeline(truncated_text, PRIORITIES)
        priority = self._extract_priority(priority_result)
        
        # Enhance priority with keyword-based rules
        priority = self._enhance_priority(text, priority)
        
        final_result = {
            "topic_tags": topic_tags,
            "sentiment": sentiment,
            "priority": priority
        }
        
        return final_result
    
    def classify_one(self, text: str) -> dict:
        """Wrapper for classify method to match app.py expectations."""
        return self.classify(text)
    
    def classify_many(self, texts: list) -> list:
        """Classify multiple texts."""
        results = []
        for text in texts:
            results.append(self.classify(text))
        return results
    
    def _map_sentiment(self, label: str, score: float) -> str:
        """Map sentiment model output to our labels."""
        mapping = {
            "negative": "Frustrated",  # Negative sentiment -> Frustrated
            "neutral": "Neutral",      # Neutral sentiment -> Neutral
            "positive": "Curious"      # Positive sentiment -> Curious
        }
        
        base_sentiment = mapping.get(label, "Neutral")
        
        # For high confidence negative sentiment, check if it's more "Angry" than "Frustrated"
        if label == "negative" and score > 0.8:
            return "Angry"  # High confidence negative = Angry
        elif label == "negative":
            return "Frustrated"  # Lower confidence negative = Frustrated
        
        return base_sentiment
    
    def _extract_topics(self, topic_result) -> list:
        """Extract relevant topics from zero-shot classification."""
        labels = topic_result['labels']
        scores = topic_result['scores']
        
        # Use higher threshold for better precision - only include topics with confidence > 0.3
        relevant_topics = []
        for label, score in zip(labels, scores):
            if score > 0.3:
                relevant_topics.append(label)
        
        # If no topics meet threshold, take the top 1-2 with highest scores
        if not relevant_topics:
            # Take top 2 topics if they have reasonable scores
            if scores[0] > 0.15:
                relevant_topics = [labels[0]]
                if len(scores) > 1 and scores[1] > 0.15:
                    relevant_topics.append(labels[1])
            else:
                relevant_topics = ["Product"]  # Default fallback
        
        return relevant_topics[:2]  # Limit to 2 most relevant topics
    
    def _enhance_topics(self, text: str, topic_tags: list) -> list:
        """Enhance topic detection with keyword-based rules for better accuracy."""
        text_lower = text.lower()
        enhanced_topics = list(topic_tags)  # Start with detected topics
        
        # Basic product questions - override false "Sensitive data" classifications
        if any(phrase in text_lower for phrase in ["what is atlan", "what is", "tell me about atlan", "explain atlan"]):
            # Remove "Sensitive data" if it was incorrectly classified
            if "Sensitive data" in enhanced_topics:
                enhanced_topics.remove("Sensitive data")
            # Add "Product" as the primary topic for basic questions
            if "Product" not in enhanced_topics:
                enhanced_topics.insert(0, "Product")
        
        # Snowflake/Connector specific detection
        if any(term in text_lower for term in ["snowflake", "snowflake", "connect", "connection", "permissions", "credentials"]):
            if "Connector" not in enhanced_topics:
                enhanced_topics.append("Connector")
            if "How-to" not in enhanced_topics:
                enhanced_topics.append("How-to")
        
        # Airflow/ETL specific detection
        if any(term in text_lower for term in ["airflow", "dag", "etl", "pipeline", "workflow"]):
            if "Lineage" not in enhanced_topics:
                enhanced_topics.append("Lineage")
            if "Connector" not in enhanced_topics:
                enhanced_topics.append("Connector")
        
        # API/SDK specific detection
        if any(term in text_lower for term in ["api", "sdk", "python", "authentication", "endpoint", "key", "programmatically"]):
            if "API/SDK" not in enhanced_topics:
                enhanced_topics.append("API/SDK")
        
        # Configuration/setup detection
        if any(term in text_lower for term in ["configure", "setup", "install", "how to", "how do", "tutorial", "guide"]):
            if "How-to" not in enhanced_topics:
                enhanced_topics.append("How-to")
        
        # Lineage specific detection
        if any(term in text_lower for term in ["lineage", "relationship", "dependency", "flow", "map", "upstream", "downstream"]):
            if "Lineage" not in enhanced_topics:
                enhanced_topics.append("Lineage")
        
        # Product features detection
        if any(term in text_lower for term in ["visual query", "query builder", "sample", "preview", "schema", "export", "not working", "broken", "issue", "problem", "bug"]):
            if "Product" not in enhanced_topics:
                enhanced_topics.append("Product")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_topics = []
        for topic in enhanced_topics:
            if topic not in seen:
                seen.add(topic)
                unique_topics.append(topic)
        
        return unique_topics[:2]  # Limit to 2 most relevant topics
    
    def _extract_priority(self, priority_result) -> str:
        """Extract priority from zero-shot classification result."""
        labels = priority_result['labels']
        scores = priority_result['scores']
        
        # Get the highest scoring priority
        best_priority = labels[0]  # BART-MNLI returns sorted by score
        best_score = scores[0]
        
        # Only return the priority if confidence is reasonable
        if best_score > 0.3:
            return best_priority
        else:
            return "P2"  # Default to medium priority if uncertain
    
    def _enhance_priority(self, text: str, priority: str) -> str:
        """Enhance priority detection with keyword-based rules."""
        text_lower = text.lower()
        
        # High priority indicators
        high_priority_terms = [
            "critical", "urgent", "outage", "down", "broken", 
            "blocked", "emergency", "asap", "immediately", "production down",
            "security", "breach", "data loss", "corruption", "system down"
        ]
        
        # Medium priority indicators  
        medium_priority_terms = [
            "important", "soon", "deadline", "project", "integration",
            "setup", "configuration", "permissions", "access", "not working properly"
        ]
        
        # Check for high priority terms
        if any(term in text_lower for term in high_priority_terms):
            return "P0"
        
        # Check for medium priority terms
        if any(term in text_lower for term in medium_priority_terms):
            return "P1"
        
        return priority  # Keep original if no enhancement
    

def load_classifier(path="models/classifier"):
    """Load the best available classifier."""
    return CustomerSupportClassifier()


def classify(text: str, tokenizer, model) -> dict:
	"""Classify text using the best available classifier."""
	return model.classify(text)




