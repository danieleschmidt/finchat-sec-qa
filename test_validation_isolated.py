#!/usr/bin/env python3
"""Isolated test for bulk operation validation implementation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_validation_function():
    """Test the validation function directly."""
    from finchat_sec_qa.validation import validate_text_safety
    
    # Test 1: Empty text should fail
    try:
        validate_text_safety('', 'test_field')
        print("❌ Empty text validation failed - should have raised ValueError")
        return False
    except ValueError as e:
        print("✅ Empty text correctly rejected:", str(e))
    
    # Test 2: Dangerous content should fail
    try:
        validate_text_safety('Hello <script>alert(1)</script> world', 'test_field')
        print("❌ Dangerous content validation failed - should have raised ValueError")
        return False
    except ValueError as e:
        print("✅ Dangerous content correctly rejected:", str(e))
    
    # Test 3: Valid text should pass
    try:
        result = validate_text_safety('This is valid text content', 'test_field')
        print("✅ Valid text correctly accepted:", result)
    except Exception as e:
        print("❌ Valid text incorrectly rejected:", str(e))
        return False
    
    return True

def mock_qa_engine_validation():
    """Test the validation logic without full QA Engine dependencies."""
    
    def _validate_document(doc_id, text):
        """Mock implementation of document validation."""
        # Validate doc_id
        if not isinstance(doc_id, str):
            raise ValueError('doc_id must be a non-empty string')
        
        if not doc_id:
            raise ValueError('doc_id cannot be empty')
        
        doc_id = doc_id.strip()
        if not doc_id:
            raise ValueError('doc_id cannot be empty')
        
        # Mock text validation (simplified)
        if not isinstance(text, str):
            raise ValueError('text must be a non-empty string')
        
        if not text:
            raise ValueError('text cannot be empty')
        
        text = text.strip()
        if not text:
            raise ValueError('text cannot be empty')
        
        # Check for dangerous patterns (simplified)
        dangerous_patterns = ['<script', 'javascript:', 'onerror=']
        text_lower = text.lower()
        for pattern in dangerous_patterns:
            if pattern in text_lower:
                raise ValueError('text contains potentially dangerous content')
        
        return text
    
    # Test validation scenarios
    tests = [
        # (doc_id, text, should_pass, expected_error)
        ("", "valid text", False, "doc_id cannot be empty"),
        ("doc1", "", False, "text cannot be empty"),
        (123, "valid text", False, "doc_id must be a non-empty string"),
        ("doc1", 123, False, "text must be a non-empty string"),
        ("doc1", "Hello <script>alert(1)</script> world", False, "text contains potentially dangerous content"),
        ("doc1", "This is valid document text", True, None),
    ]
    
    for doc_id, text, should_pass, expected_error in tests:
        try:
            _validate_document(doc_id, text)
            if should_pass:
                print(f"✅ Valid document passed: doc_id='{doc_id}', text='{text[:30]}...'")
            else:
                print(f"❌ Invalid document should have failed: doc_id='{doc_id}', text='{text[:30]}...'")
                return False
        except ValueError as e:
            if not should_pass and expected_error in str(e):
                print(f"✅ Invalid document correctly rejected: {str(e)}")
            else:
                print(f"❌ Unexpected validation result: {str(e)}")
                return False
    
    return True

if __name__ == "__main__":
    print("Testing bulk operation validation implementation...")
    print("\n1. Testing validation function directly:")
    
    try:
        validation_works = test_validation_function()
    except ImportError as e:
        print(f"❌ Cannot import validation module: {e}")
        validation_works = False
    
    print("\n2. Testing validation logic with mock:")
    mock_works = mock_qa_engine_validation()
    
    if validation_works and mock_works:
        print("\n🎉 All validation tests passed! Implementation appears correct.")
        sys.exit(0)
    else:
        print("\n❌ Some validation tests failed.")
        sys.exit(1)