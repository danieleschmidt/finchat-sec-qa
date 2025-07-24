"""Comprehensive unit tests for risk intelligence module."""
import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from finchat_sec_qa.risk_intelligence import RiskAnalyzer, RiskAssessment


class TestRiskAssessment(unittest.TestCase):
    """Test RiskAssessment dataclass."""
    
    def test_risk_assessment_creation(self):
        """Test creating RiskAssessment with required fields."""
        assessment = RiskAssessment(text="Test text", sentiment=0.5)
        self.assertEqual(assessment.text, "Test text")
        self.assertEqual(assessment.sentiment, 0.5)
        self.assertEqual(assessment.flags, [])  # Default empty list
    
    def test_risk_assessment_with_flags(self):
        """Test creating RiskAssessment with flags."""
        flags = ["litigation", "financial"]
        assessment = RiskAssessment(text="Test", sentiment=-0.3, flags=flags)
        self.assertEqual(assessment.text, "Test")
        self.assertEqual(assessment.sentiment, -0.3)
        self.assertEqual(assessment.flags, flags)
    
    def test_risk_assessment_fields_accessible(self):
        """Test that all fields are accessible."""
        assessment = RiskAssessment(
            text="Company faces legal challenges",
            sentiment=-0.7,
            flags=["litigation"]
        )
        # Should be able to access all fields
        self.assertIsInstance(assessment.text, str)
        self.assertIsInstance(assessment.sentiment, float)
        self.assertIsInstance(assessment.flags, list)


class TestRiskAnalyzer(unittest.TestCase):
    """Test RiskAnalyzer functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = RiskAnalyzer()
    
    def test_risk_analyzer_initialization_default(self):
        """Test RiskAnalyzer initializes with default keyword map."""
        analyzer = RiskAnalyzer()
        self.assertIsNotNone(analyzer.analyzer)  # SentimentIntensityAnalyzer
        self.assertIsNotNone(analyzer.keyword_map)
        
        # Check default categories exist
        self.assertIn("litigation", analyzer.keyword_map)
        self.assertIn("financial", analyzer.keyword_map)
        self.assertIn("operational", analyzer.keyword_map)
    
    def test_risk_analyzer_custom_keyword_map(self):
        """Test RiskAnalyzer with custom keyword map."""
        custom_map = {
            "security": ["hack", "breach", "cybersecurity"],
            "regulatory": ["compliance", "violation", "penalty"]
        }
        analyzer = RiskAnalyzer(keyword_map=custom_map)
        self.assertEqual(analyzer.keyword_map, custom_map)
    
    def test_assess_positive_sentiment(self):
        """Test assessment of positive sentiment text."""
        text = "The company exceeded expectations with record profits and strong growth."
        result = self.analyzer.assess(text)
        
        self.assertIsInstance(result, RiskAssessment)
        self.assertEqual(result.text, text)
        self.assertGreater(result.sentiment, 0, "Should detect positive sentiment")
        self.assertEqual(result.flags, [], "Should not detect risk flags in positive text")
    
    def test_assess_negative_sentiment(self):
        """Test assessment of negative sentiment text."""
        text = "The company reported significant losses and declining market share."
        result = self.analyzer.assess(text)
        
        self.assertIsInstance(result, RiskAssessment)
        self.assertEqual(result.text, text)
        self.assertLess(result.sentiment, 0, "Should detect negative sentiment")
    
    def test_assess_neutral_sentiment(self):
        """Test assessment of neutral sentiment text."""
        text = "The company released its quarterly report today."
        result = self.analyzer.assess(text)
        
        self.assertIsInstance(result, RiskAssessment)
        self.assertEqual(result.text, text)
        # Sentiment should be close to 0 (neutral)
        self.assertGreaterEqual(result.sentiment, -0.2)
        self.assertLessEqual(result.sentiment, 0.2)
    
    def test_litigation_risk_detection(self):
        """Test detection of litigation risks."""
        test_cases = [
            "The company is facing a major lawsuit from shareholders.",
            "New litigation has been filed against the corporation.",
            "Legal proceedings have begun regarding patent infringement."
        ]
        
        for text in test_cases:
            with self.subTest(text=text):
                result = self.analyzer.assess(text)
                self.assertIn("litigation", result.flags, f"Should detect litigation risk in: {text}")
    
    def test_financial_risk_detection(self):
        """Test detection of financial risks."""
        test_cases = [
            "The company is considering bankruptcy protection.",
            "Liquidity concerns have emerged due to cash flow issues.",
            "The firm defaulted on its debt obligations.",
            "Insolvency proceedings may be necessary."
        ]
        
        for text in test_cases:
            with self.subTest(text=text):
                result = self.analyzer.assess(text)
                self.assertIn("financial", result.flags, f"Should detect financial risk in: {text}")
    
    def test_operational_risk_detection(self):
        """Test detection of operational risks."""
        test_cases = [
            "The factory shutdown has impacted production capacity.",
            "Workers went on strike affecting operations.",
            "A major product recall was announced today.",
            "Supply chain disruption continues to affect deliveries."
        ]
        
        for text in test_cases:
            with self.subTest(text=text):
                result = self.analyzer.assess(text)
                self.assertIn("operational", result.flags, f"Should detect operational risk in: {text}")
    
    def test_multiple_risk_categories(self):
        """Test detection of multiple risk categories in single text."""
        text = "The company faces both a lawsuit over bankruptcy proceedings and operational disruption."
        result = self.analyzer.assess(text)
        
        # Should detect multiple risk types
        self.assertIn("litigation", result.flags, "Should detect litigation risk")
        self.assertIn("financial", result.flags, "Should detect financial risk")
        self.assertIn("operational", result.flags, "Should detect operational risk")
        self.assertEqual(len(result.flags), 3, "Should detect all three risk categories")
    
    def test_case_insensitive_detection(self):
        """Test that risk detection is case insensitive."""
        test_cases = [
            "LAWSUIT filed against company",
            "Lawsuit Filed Against Company", 
            "lawsuit filed against company",
            "LaWsUiT filed against company"
        ]
        
        for text in test_cases:
            with self.subTest(text=text):
                result = self.analyzer.assess(text)
                self.assertIn("litigation", result.flags, f"Should detect litigation regardless of case: {text}")
    
    def test_partial_word_matching(self):
        """Test that risk detection works with partial word matching."""
        # The current implementation uses 'word in text' which should match substrings
        text = "The company faces upcoming litigation proceedings."
        result = self.analyzer.assess(text)
        self.assertIn("litigation", result.flags, "Should detect 'litigation' in text")
    
    def test_no_false_positives(self):
        """Test that normal business text doesn't trigger false risk flags."""
        clean_texts = [
            "The company announced a new product launch.",
            "Quarterly earnings exceeded analyst expectations.",
            "The board approved the dividend increase.",
            "Management provided guidance for next year."
        ]
        
        for text in clean_texts:
            with self.subTest(text=text):
                result = self.analyzer.assess(text)
                self.assertEqual(result.flags, [], f"Should not detect false risks in: {text}")
    
    def test_sentiment_score_range(self):
        """Test that sentiment scores are within expected range."""
        texts = [
            "Excellent performance and outstanding results",  # Very positive
            "Good growth and solid fundamentals",            # Positive
            "The company reported quarterly results",        # Neutral
            "Challenges and concerns about performance",     # Negative
            "Terrible losses and catastrophic failures"     # Very negative
        ]
        
        for text in texts:
            with self.subTest(text=text):
                result = self.analyzer.assess(text)
                self.assertGreaterEqual(result.sentiment, -1.0, "Sentiment should be >= -1.0")
                self.assertLessEqual(result.sentiment, 1.0, "Sentiment should be <= 1.0")
                self.assertIsInstance(result.sentiment, float, "Sentiment should be float")
    
    def test_empty_text_handling(self):
        """Test handling of empty or whitespace text."""
        empty_cases = ["", "   ", "\n\t  \n"]
        
        for text in empty_cases:
            with self.subTest(text=repr(text)):
                result = self.analyzer.assess(text)
                self.assertEqual(result.text, text)
                self.assertIsInstance(result.sentiment, float)
                self.assertEqual(result.flags, [])
    
    def test_assess_returns_correct_type(self):
        """Test that assess method always returns RiskAssessment."""
        test_texts = [
            "Normal business text",
            "Lawsuit and bankruptcy issues",
            "",
            "123 numbers only",
            "Special chars !@#$%^&*()"
        ]
        
        for text in test_texts:
            with self.subTest(text=text):
                result = self.analyzer.assess(text)
                self.assertIsInstance(result, RiskAssessment)
                self.assertIsInstance(result.text, str)
                self.assertIsInstance(result.sentiment, float)
                self.assertIsInstance(result.flags, list)
    
    def test_keyword_map_modification_isolation(self):
        """Test that modifying keyword map doesn't affect other instances."""
        analyzer1 = RiskAnalyzer()
        analyzer2 = RiskAnalyzer()
        
        # Modify first analyzer's keyword map
        analyzer1.keyword_map["custom"] = ["test", "example"]
        
        # Second analyzer should not be affected
        self.assertNotIn("custom", analyzer2.keyword_map)
        
        # Verify they work independently
        text = "This is a test example"
        result1 = analyzer1.assess(text)
        result2 = analyzer2.assess(text)
        
        self.assertIn("custom", result1.flags)
        self.assertNotIn("custom", result2.flags)


if __name__ == '__main__':
    unittest.main()