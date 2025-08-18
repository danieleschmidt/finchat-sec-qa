from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


@dataclass
class RiskAssessment:
    """Result of analyzing a passage for sentiment and risk."""

    text: str
    sentiment: float
    flags: List[str] = field(default_factory=list)
    risk_level: str = "low"
    risk_factors: List[str] = field(default_factory=list)


class RiskAnalyzer:
    """Perform sentiment analysis and risk flag tagging."""

    def __init__(self, keyword_map: Dict[str, List[str]] | None = None) -> None:
        self.analyzer = SentimentIntensityAnalyzer()
        self.keyword_map = keyword_map or {
            "litigation": ["lawsuit", "litigation", "legal proceeding"],
            "financial": ["bankruptcy", "insolvency", "default", "liquidity"],
            "operational": ["shutdown", "strike", "recall", "disruption"],
        }

    def assess(self, text: str) -> RiskAssessment:
        """Return sentiment score and detected risk categories."""
        score = self.analyzer.polarity_scores(text)["compound"]
        lower = text.lower()
        flags = [
            category
            for category, words in self.keyword_map.items()
            if any(word in lower for word in words)
        ]
        
        # Determine risk level
        risk_level = "low"
        if score < -0.5 or len(flags) > 2:
            risk_level = "high"
        elif score < -0.2 or len(flags) > 0:
            risk_level = "medium"
        
        return RiskAssessment(
            text=text, 
            sentiment=score, 
            flags=flags,
            risk_level=risk_level,
            risk_factors=flags
        )
    
    def analyze_text(self, text: str) -> RiskAssessment:
        """Alias for assess method to match expected interface."""
        return self.assess(text)
