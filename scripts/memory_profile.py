#!/usr/bin/env python3
"""Memory profiling script for FinChat SEC QA."""

import argparse
import sys
import time
from memory_profiler import profile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from finchat_sec_qa.qa_engine import QAEngine


@profile
def memory_intensive_operations():
    """Run memory-intensive operations for profiling."""
    print("Starting memory profiling...")
    
    # Initialize QA engine
    qa_engine = QAEngine()
    
    # Simulate large document processing
    large_text = "This is a test document. " * 10000
    
    for i in range(10):
        print(f"Processing iteration {i+1}/10")
        
        # Process document chunks
        chunks = qa_engine._chunk_text(large_text)
        
        # Generate embeddings (simulated)
        time.sleep(0.1)
        
        # Simulate query processing
        result = qa_engine.query(
            question="What are the main risk factors?",
            context=large_text[:5000]
        )
        
        print(f"Processed {len(chunks)} chunks")
    
    print("Memory profiling completed")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Memory profiling for FinChat")
    parser.add_argument("--output", default="memory-profile.txt", 
                       help="Output file for memory profile")
    
    args = parser.parse_args()
    
    print(f"Running memory profiling, output will be saved to {args.output}")
    
    # Run the profiling
    memory_intensive_operations()


if __name__ == "__main__":
    main()