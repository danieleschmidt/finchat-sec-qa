#!/usr/bin/env python3
"""Test chunking logic refactoring functionality."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_helper_methods_exist():
    """Test that helper methods exist."""
    try:
        from finchat_sec_qa.qa_engine import FinancialQAEngine
        
        engine = FinancialQAEngine()
        
        # Check helper methods exist
        methods = ['_is_single_chunk', '_create_overlapping_chunks', '_create_next_chunk', 
                  '_find_sentence_boundary', '_create_chunk_at_boundary', '_create_chunk_at_position']
        
        for method in methods:
            if not hasattr(engine, method):
                print(f"❌ Missing method: {method}")
                return False
        
        print("✅ All helper methods exist")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def test_single_chunk_detection():
    """Test single chunk detection."""
    try:
        from finchat_sec_qa.qa_engine import FinancialQAEngine
        
        engine = FinancialQAEngine()
        
        # Test short text
        short_text = "Short text."
        result = engine._is_single_chunk(short_text)
        assert isinstance(result, bool)
        print(f"✅ Single chunk detection works: '{short_text}' -> {result}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def test_chunking_functionality():
    """Test basic chunking functionality."""
    try:
        from finchat_sec_qa.qa_engine import FinancialQAEngine
        
        engine = FinancialQAEngine()
        
        # Test various text scenarios
        test_cases = [
            "Short text that should be a single chunk.",
            "Medium text with sentences. Multiple sentences here. More content follows.",
            "A" * 50 + ". " + "B" * 50 + ". " + "C" * 50 + ".",  # Text with boundaries
        ]
        
        for i, text in enumerate(test_cases):
            try:
                chunks = engine._chunk_text(text)
                
                # Validate basic properties
                assert isinstance(chunks, list)
                assert len(chunks) > 0
                
                for chunk in chunks:
                    assert isinstance(chunk, tuple)
                    assert len(chunk) == 3
                    assert isinstance(chunk[0], str)
                    assert isinstance(chunk[1], int)
                    assert isinstance(chunk[2], int)
                
                print(f"✅ Test case {i+1} passed: {len(chunks)} chunks")
                
            except Exception as e:
                print(f"❌ Test case {i+1} failed: {e}")
                return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

if __name__ == "__main__":
    print("Testing chunking logic refactoring...")
    
    test1 = test_helper_methods_exist()
    test2 = test_single_chunk_detection()
    test3 = test_chunking_functionality()
    
    if test1 and test2 and test3:
        print("\n🎉 All chunking refactoring tests passed! Complex method successfully split.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed.")
        sys.exit(1)