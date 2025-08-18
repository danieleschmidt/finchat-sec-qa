"""
Comprehensive Validation System - Generation 2: MAKE IT ROBUST
TERRAGON SDLC v4.0 - Autonomous Execution Phase

Features:
- Multi-layer input validation and sanitization
- Real-time security threat detection
- Data integrity verification
- Autonomous anomaly detection
- Comprehensive schema validation
- Performance-aware validation
"""

from __future__ import annotations

import logging
import re
import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Callable, Type
from enum import Enum
import json
import html
import urllib.parse

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    PARANOID = "paranoid"


class ThreatLevel(Enum):
    """Security threat levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of validation process."""
    is_valid: bool
    original_value: Any
    sanitized_value: Any
    validation_level: ValidationLevel
    threat_level: ThreatLevel
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0


@dataclass
class ValidationRule:
    """Individual validation rule."""
    name: str
    validator_func: Callable[[Any], bool]
    sanitizer_func: Optional[Callable[[Any], Any]] = None
    error_message: str = ""
    threat_level: ThreatLevel = ThreatLevel.LOW


class ComprehensiveValidationSystem:
    """
    Generation 2: Comprehensive validation system with security and performance focus.
    
    Features:
    - Multi-layer validation (syntax, semantic, security)
    - Real-time threat detection
    - Autonomous anomaly detection
    - Performance-optimized validation
    - Comprehensive sanitization
    """
    
    def __init__(self, default_level: ValidationLevel = ValidationLevel.STANDARD):
        self.default_level = default_level
        self.validation_rules: Dict[str, List[ValidationRule]] = {}
        self.threat_patterns: List[Dict[str, Any]] = []
        self.validation_cache: Dict[str, ValidationResult] = {}
        self.validation_stats: Dict[str, Any] = {
            'total_validations': 0,
            'threats_detected': 0,
            'cache_hits': 0,
            'avg_processing_time': 0.0
        }
        
        self._initialize_security_patterns()
        self._initialize_standard_rules()
        
        logger.info("Comprehensive validation system initialized")
    
    def _initialize_security_patterns(self):
        """Initialize security threat detection patterns."""
        self.threat_patterns = [
            {
                'name': 'sql_injection',
                'pattern': r'(\bSELECT\b|\bINSERT\b|\bUPDATE\b|\bDELETE\b|\bDROP\b|\bUNION\b|--|;|\bOR\b\s+\d+\s*=\s*\d+)',
                'threat_level': ThreatLevel.HIGH,
                'description': 'Potential SQL injection attack'
            },
            {
                'name': 'xss_attack',
                'pattern': r'(<script[^>]*>|javascript:|on\w+\s*=|<iframe|<object|<embed)',
                'threat_level': ThreatLevel.HIGH,
                'description': 'Potential XSS attack'
            },
            {
                'name': 'path_traversal',
                'pattern': r'(\.\./|\.\.\\|%2e%2e%2f|%252e%252e%252f)',
                'threat_level': ThreatLevel.MEDIUM,
                'description': 'Potential path traversal attack'
            },
            {
                'name': 'command_injection',
                'pattern': r'(\||\&|\;|\$\(|\`|system\(|exec\(|eval\(|passthru\()',
                'threat_level': ThreatLevel.HIGH,
                'description': 'Potential command injection'
            },
            {
                'name': 'ldap_injection',
                'pattern': r'(\(\||\)\(|\*\)|\(\&)',
                'threat_level': ThreatLevel.MEDIUM,
                'description': 'Potential LDAP injection'
            },
            {
                'name': 'xxe_attack',
                'pattern': r'(<!ENTITY|SYSTEM|file://|http://|ftp://)',
                'threat_level': ThreatLevel.HIGH,
                'description': 'Potential XXE attack'
            },
            {
                'name': 'template_injection',
                'pattern': r'(\{\{.*\}\}|\$\{.*\})',
                'threat_level': ThreatLevel.HIGH,
                'description': 'Potential template injection'
            },
            {
                'name': 'jndi_injection',
                'pattern': r'(\$\{jndi:|ldap://|dns://)',
                'threat_level': ThreatLevel.CRITICAL,
                'description': 'Potential JNDI injection'
            }
        ]
    
    def _initialize_standard_rules(self):
        """Initialize standard validation rules."""
        
        # String validation rules
        string_rules = [
            ValidationRule(
                name="max_length",
                validator_func=lambda x: len(str(x)) <= 10000,
                sanitizer_func=lambda x: str(x)[:10000],
                error_message="String exceeds maximum length",
                threat_level=ThreatLevel.LOW
            ),
            ValidationRule(
                name="no_null_bytes",
                validator_func=lambda x: '\x00' not in str(x),
                sanitizer_func=lambda x: str(x).replace('\x00', ''),
                error_message="String contains null bytes",
                threat_level=ThreatLevel.MEDIUM
            ),
            ValidationRule(
                name="printable_chars",
                validator_func=lambda x: all(ord(c) >= 32 or c in '\t\n\r' for c in str(x)),
                sanitizer_func=lambda x: ''.join(c for c in str(x) if ord(c) >= 32 or c in '\t\n\r'),
                error_message="String contains non-printable characters",
                threat_level=ThreatLevel.LOW
            )
        ]
        
        # Financial query rules
        financial_rules = [
            ValidationRule(
                name="reasonable_query_length",
                validator_func=lambda x: 10 <= len(str(x)) <= 1000,
                error_message="Query length outside reasonable bounds",
                threat_level=ThreatLevel.LOW
            ),
            ValidationRule(
                name="contains_financial_terms",
                validator_func=lambda x: any(term in str(x).lower() for term in 
                    ['revenue', 'profit', 'cash', 'debt', 'risk', 'earnings', 'income', 'assets', 'liability']),
                error_message="Query doesn't contain recognizable financial terms",
                threat_level=ThreatLevel.LOW
            )
        ]
        
        # Ticker validation rules
        ticker_rules = [
            ValidationRule(
                name="valid_ticker_format",
                validator_func=lambda x: re.match(r'^[A-Z]{1,5}$', str(x).upper()) is not None,
                sanitizer_func=lambda x: str(x).upper().strip(),
                error_message="Invalid ticker symbol format",
                threat_level=ThreatLevel.LOW
            )
        ]
        
        self.validation_rules = {
            'string': string_rules,
            'financial_query': financial_rules,
            'ticker': ticker_rules
        }
    
    def validate(self, 
                value: Any, 
                data_type: str = 'string',
                validation_level: Optional[ValidationLevel] = None,
                custom_rules: Optional[List[ValidationRule]] = None) -> ValidationResult:
        """
        Comprehensive validation with threat detection and sanitization.
        
        Args:
            value: Value to validate
            data_type: Type of data for rule selection
            validation_level: Validation strictness level
            custom_rules: Additional custom validation rules
            
        Returns:
            Comprehensive validation result
        """
        start_time = time.time()
        level = validation_level or self.default_level
        
        # Check cache first
        cache_key = self._generate_cache_key(value, data_type, level)
        if cache_key in self.validation_cache:
            self.validation_stats['cache_hits'] += 1
            cached_result = self.validation_cache[cache_key]
            logger.debug(f"Validation cache hit for {data_type}")
            return cached_result
        
        # Initialize result
        result = ValidationResult(
            is_valid=True,
            original_value=value,
            sanitized_value=value,
            validation_level=level,
            threat_level=ThreatLevel.NONE
        )
        
        try:
            # Step 1: Basic type and format validation
            result = self._validate_basic_format(result, data_type)
            
            # Step 2: Security threat detection
            result = self._detect_security_threats(result)
            
            # Step 3: Apply validation rules
            result = self._apply_validation_rules(result, data_type, custom_rules)
            
            # Step 4: Advanced validation based on level
            if level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                result = self._advanced_validation(result, data_type)
            
            # Step 5: Anomaly detection for paranoid level
            if level == ValidationLevel.PARANOID:
                result = self._anomaly_detection(result, data_type)
            
            # Step 6: Final sanitization
            result = self._final_sanitization(result)
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            result.is_valid = False
            result.issues.append(f"Validation system error: {str(e)}")
            result.threat_level = ThreatLevel.HIGH
        
        # Record metrics
        processing_time = (time.time() - start_time) * 1000
        result.processing_time_ms = processing_time
        self._update_stats(result)
        
        # Cache result
        self.validation_cache[cache_key] = result
        
        # Cleanup cache if too large
        if len(self.validation_cache) > 1000:
            self._cleanup_cache()
        
        logger.debug(f"Validation completed for {data_type} in {processing_time:.2f}ms")
        return result
    
    def _validate_basic_format(self, result: ValidationResult, data_type: str) -> ValidationResult:
        """Basic format validation."""
        value = result.original_value
        
        if value is None:
            result.is_valid = False
            result.issues.append("Value is None")
            return result
        
        # Convert to string for analysis
        str_value = str(value)
        
        # Basic length check
        if len(str_value) > 100000:  # 100KB limit
            result.is_valid = False
            result.issues.append("Value exceeds maximum size limit")
            result.threat_level = self._escalate_threat_level(result.threat_level, ThreatLevel.MEDIUM)
        
        # Check for obviously malicious patterns
        if len(str_value) > 1000 and str_value.count('<') > 10:
            result.warnings.append("High HTML tag density detected")
            result.threat_level = self._escalate_threat_level(result.threat_level, ThreatLevel.LOW)
        
        return result
    
    def _detect_security_threats(self, result: ValidationResult) -> ValidationResult:
        """Detect security threats using pattern matching."""
        str_value = str(result.original_value).lower()
        
        for pattern_info in self.threat_patterns:
            pattern = pattern_info['pattern']
            if re.search(pattern, str_value, re.IGNORECASE):
                threat_detected = pattern_info['name']
                threat_level = pattern_info['threat_level']
                description = pattern_info['description']
                
                result.threat_level = self._escalate_threat_level(result.threat_level, threat_level)
                result.issues.append(f"Security threat detected: {description}")
                result.metadata[f'threat_{threat_detected}'] = True
                
                # Mark as invalid for high threats
                if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                    result.is_valid = False
                
                logger.warning(f"Security threat detected: {threat_detected} in validation")
        
        return result
    
    def _apply_validation_rules(self, 
                              result: ValidationResult, 
                              data_type: str,
                              custom_rules: Optional[List[ValidationRule]]) -> ValidationResult:
        """Apply validation rules for the data type."""
        
        # Get rules for data type
        rules = self.validation_rules.get(data_type, [])
        if custom_rules:
            rules.extend(custom_rules)
        
        for rule in rules:
            try:
                # Apply validator
                if not rule.validator_func(result.sanitized_value):
                    result.is_valid = False
                    result.issues.append(rule.error_message or f"Validation rule '{rule.name}' failed")
                    result.threat_level = self._escalate_threat_level(result.threat_level, rule.threat_level)
                
                # Apply sanitizer if available
                if rule.sanitizer_func:
                    result.sanitized_value = rule.sanitizer_func(result.sanitized_value)
                
            except Exception as e:
                logger.warning(f"Validation rule '{rule.name}' failed with error: {e}")
                result.warnings.append(f"Validation rule '{rule.name}' encountered an error")
        
        return result
    
    def _advanced_validation(self, result: ValidationResult, data_type: str) -> ValidationResult:
        """Advanced validation for strict and paranoid levels."""
        
        if data_type == 'financial_query':
            # Advanced financial query validation
            query = str(result.sanitized_value).lower()
            
            # Check for balanced parentheses
            if query.count('(') != query.count(')'):
                result.warnings.append("Unbalanced parentheses in query")
            
            # Check for reasonable word count
            word_count = len(query.split())
            if word_count < 3:
                result.warnings.append("Query seems too short to be meaningful")
            elif word_count > 100:
                result.warnings.append("Query seems unusually long")
            
            # Check for financial context
            financial_terms = ['revenue', 'profit', 'earnings', 'cash', 'debt', 'risk', 'growth', 'margin']
            financial_score = sum(1 for term in financial_terms if term in query)
            
            if financial_score == 0:
                result.warnings.append("Query lacks clear financial context")
            
            result.metadata['financial_relevance_score'] = financial_score / len(financial_terms)
        
        elif data_type == 'ticker':
            # Advanced ticker validation
            ticker = str(result.sanitized_value).upper()
            
            # Check against common ticker patterns
            if len(ticker) < 1 or len(ticker) > 5:
                result.is_valid = False
                result.issues.append("Ticker length outside valid range")
            
            # Check for invalid characters
            if not ticker.isalpha():
                result.is_valid = False
                result.issues.append("Ticker contains non-alphabetic characters")
        
        return result
    
    def _anomaly_detection(self, result: ValidationResult, data_type: str) -> ValidationResult:
        """Anomaly detection for paranoid validation level."""
        
        # Statistical anomaly detection
        value_str = str(result.sanitized_value)
        
        # Character frequency analysis
        char_freq = {}
        for char in value_str:
            char_freq[char] = char_freq.get(char, 0) + 1
        
        # Check for character repetition anomalies
        if len(value_str) > 10:
            max_freq = max(char_freq.values())
            if max_freq > len(value_str) * 0.5:  # More than 50% same character
                result.warnings.append("High character repetition detected")
                result.threat_level = self._escalate_threat_level(result.threat_level, ThreatLevel.LOW)
        
        # Entropy check for randomness
        if len(value_str) > 20:
            entropy = self._calculate_entropy(value_str)
            if entropy < 1.0:  # Very low entropy
                result.warnings.append("Low entropy detected - possible pattern or attack")
            elif entropy > 4.5:  # Very high entropy
                result.warnings.append("High entropy detected - possible encoded content")
        
        # Pattern repetition check
        if len(value_str) > 50:
            # Look for repeated substrings
            for length in [3, 4, 5]:
                substrings = {}
                for i in range(len(value_str) - length + 1):
                    substr = value_str[i:i+length]
                    substrings[substr] = substrings.get(substr, 0) + 1
                
                max_repetitions = max(substrings.values()) if substrings else 0
                if max_repetitions > 5:
                    result.warnings.append(f"High pattern repetition detected (length {length})")
                    break
        
        return result
    
    def _final_sanitization(self, result: ValidationResult) -> ValidationResult:
        """Final sanitization pass."""
        
        if isinstance(result.sanitized_value, str):
            # HTML entity encoding for safety
            result.sanitized_value = html.escape(result.sanitized_value)
            
            # URL encoding for special characters if needed
            if any(char in result.sanitized_value for char in ['<', '>', '"', "'"]):
                result.metadata['html_escaped'] = True
            
            # Trim whitespace
            result.sanitized_value = result.sanitized_value.strip()
            
            # Remove control characters except common ones
            result.sanitized_value = ''.join(
                char for char in result.sanitized_value 
                if ord(char) >= 32 or char in '\t\n\r'
            )
        
        return result
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text."""
        if not text:
            return 0.0
        
        # Count character frequencies
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Calculate entropy
        text_length = len(text)
        entropy = 0.0
        
        for count in char_counts.values():
            probability = count / text_length
            if probability > 0:
                import math
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def _generate_cache_key(self, value: Any, data_type: str, level: ValidationLevel) -> str:
        """Generate cache key for validation result."""
        value_hash = hashlib.md5(str(value).encode()).hexdigest()
        return f"{data_type}_{level.value}_{value_hash}"
    
    def _update_stats(self, result: ValidationResult):
        """Update validation statistics."""
        self.validation_stats['total_validations'] += 1
        
        if result.threat_level != ThreatLevel.NONE:
            self.validation_stats['threats_detected'] += 1
        
        # Update average processing time
        current_avg = self.validation_stats['avg_processing_time']
        total_validations = self.validation_stats['total_validations']
        
        self.validation_stats['avg_processing_time'] = (
            (current_avg * (total_validations - 1) + result.processing_time_ms) / total_validations
        )
    
    def _cleanup_cache(self):
        """Clean up validation cache when it gets too large."""
        # Keep only the most recent 500 entries
        cache_items = list(self.validation_cache.items())
        self.validation_cache = dict(cache_items[-500:])
        logger.debug("Validation cache cleaned up")
    
    def add_custom_rule(self, data_type: str, rule: ValidationRule):
        """Add custom validation rule for a data type."""
        if data_type not in self.validation_rules:
            self.validation_rules[data_type] = []
        
        self.validation_rules[data_type].append(rule)
        logger.info(f"Added custom validation rule '{rule.name}' for type '{data_type}'")
    
    def add_threat_pattern(self, name: str, pattern: str, threat_level: ThreatLevel, description: str):
        """Add custom threat detection pattern."""
        self.threat_patterns.append({
            'name': name,
            'pattern': pattern,
            'threat_level': threat_level,
            'description': description
        })
        logger.info(f"Added custom threat pattern: {name}")
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get comprehensive validation statistics."""
        return {
            **self.validation_stats,
            'cache_size': len(self.validation_cache),
            'threat_patterns_count': len(self.threat_patterns),
            'validation_rules_count': sum(len(rules) for rules in self.validation_rules.values()),
            'timestamp': datetime.now().isoformat()
        }
    
    def _escalate_threat_level(self, current: ThreatLevel, new: ThreatLevel) -> ThreatLevel:
        """Escalate threat level to the higher of current and new."""
        threat_order = {
            ThreatLevel.NONE: 0,
            ThreatLevel.LOW: 1,
            ThreatLevel.MEDIUM: 2,
            ThreatLevel.HIGH: 3,
            ThreatLevel.CRITICAL: 4
        }
        
        if threat_order[new] > threat_order[current]:
            return new
        return current
    
    def validate_batch(self, 
                      values: List[Any], 
                      data_type: str = 'string',
                      validation_level: Optional[ValidationLevel] = None) -> List[ValidationResult]:
        """Validate multiple values efficiently."""
        results = []
        
        for value in values:
            result = self.validate(value, data_type, validation_level)
            results.append(result)
        
        logger.info(f"Batch validation completed: {len(values)} items")
        return results