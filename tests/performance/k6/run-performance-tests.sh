#!/bin/bash

# Performance Test Suite Runner for FinChat SEC Q&A
# This script runs the complete performance testing suite

set -e

# Configuration
BASE_URL=${BASE_URL:-"http://localhost:8000"}
API_TOKEN=${API_TOKEN:-"test-token"}
RESULTS_DIR="./results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if k6 is installed
check_k6() {
    if ! command -v k6 &> /dev/null; then
        print_error "k6 is not installed. Please install k6 first."
        echo "Installation instructions:"
        echo "  Ubuntu/Debian: sudo apt-get install k6"
        echo "  macOS: brew install k6"
        echo "  Docker: Use 'docker run --rm -i grafana/k6:latest'"
        exit 1
    fi
    print_success "k6 is installed: $(k6 version)"
}

# Check if application is running
check_application() {
    print_status "Checking if application is running at $BASE_URL..."
    
    if curl -s -f "$BASE_URL/health" > /dev/null 2>&1; then
        print_success "Application is running and healthy"
    else
        print_error "Application is not running or not healthy at $BASE_URL"
        print_warning "Please start the application first:"
        echo "  python -m finchat_sec_qa.server"
        echo "  or docker-compose up"
        exit 1
    fi
}

# Create results directory
create_results_dir() {
    mkdir -p "$RESULTS_DIR"
    print_status "Results will be saved to: $RESULTS_DIR"
}

# Run smoke test
run_smoke_test() {
    print_status "Running smoke test..."
    
    local output_file="$RESULTS_DIR/smoke-test-$TIMESTAMP.json"
    
    if BASE_URL="$BASE_URL" API_TOKEN="$API_TOKEN" k6 run \
        --out json="$output_file" \
        smoke-test.js; then
        print_success "Smoke test completed successfully"
        echo "Results: $output_file"
    else
        print_error "Smoke test failed"
        return 1
    fi
}

# Run load test
run_load_test() {
    print_status "Running load test..."
    
    local output_file="$RESULTS_DIR/load-test-$TIMESTAMP.json"
    
    if BASE_URL="$BASE_URL" API_TOKEN="$API_TOKEN" k6 run \
        --out json="$output_file" \
        load-test.js; then
        print_success "Load test completed successfully"
        echo "Results: $output_file"
    else
        print_error "Load test failed"
        return 1
    fi
}

# Run stress test
run_stress_test() {
    print_status "Running stress test..."
    print_warning "This test will push the system to its limits"
    
    local output_file="$RESULTS_DIR/stress-test-$TIMESTAMP.json"
    
    if BASE_URL="$BASE_URL" API_TOKEN="$API_TOKEN" k6 run \
        --out json="$output_file" \
        stress-test.js; then
        print_success "Stress test completed"
        echo "Results: $output_file"
        
        # Give system time to recover
        print_status "Allowing system to recover..."
        sleep 10
    else
        print_warning "Stress test may have pushed system beyond limits (this is expected)"
        return 0  # Don't fail the entire suite for stress test failures
    fi
}

# Generate summary report
generate_summary() {
    print_status "Generating test summary..."
    
    local summary_file="$RESULTS_DIR/summary-$TIMESTAMP.txt"
    
    cat > "$summary_file" << EOF
Performance Test Suite Summary
==============================
Timestamp: $(date)
Base URL: $BASE_URL
API Token: $API_TOKEN

Test Results:
- Smoke Test: $smoke_result
- Load Test: $load_result  
- Stress Test: $stress_result

Results Directory: $RESULTS_DIR

For detailed analysis, examine the JSON result files.
EOF

    print_success "Summary saved to: $summary_file"
    cat "$summary_file"
}

# Main execution
main() {
    echo "======================================"
    echo "FinChat SEC Q&A Performance Test Suite"
    echo "======================================"
    echo ""
    
    # Pre-flight checks
    check_k6
    check_application
    create_results_dir
    
    # Test execution tracking
    smoke_result="PENDING"
    load_result="PENDING"
    stress_result="PENDING"
    
    echo ""
    print_status "Starting performance test suite..."
    echo ""
    
    # Run tests in sequence
    if run_smoke_test; then
        smoke_result="PASSED"
        
        if run_load_test; then
            load_result="PASSED"
            
            # Only run stress test if load test passes
            read -p "Run stress test? This will push the system to its limits (y/N): " -n 1 -r
            echo ""
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                if run_stress_test; then
                    stress_result="PASSED"
                else
                    stress_result="FAILED"
                fi
            else
                stress_result="SKIPPED"
            fi
        else
            load_result="FAILED"
            print_warning "Skipping stress test due to load test failure"
            stress_result="SKIPPED"
        fi
    else
        smoke_result="FAILED"
        print_error "Smoke test failed - skipping remaining tests"
        load_result="SKIPPED"
        stress_result="SKIPPED"
    fi
    
    echo ""
    generate_summary
    
    echo ""
    if [ "$smoke_result" = "PASSED" ] && [ "$load_result" = "PASSED" ]; then
        print_success "Performance test suite completed successfully!"
        exit 0
    else
        print_error "Some tests failed - review results for details"
        exit 1
    fi
}

# Script help
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Environment Variables:"
    echo "  BASE_URL    Application base URL (default: http://localhost:8000)"
    echo "  API_TOKEN   API authentication token (default: test-token)"
    echo ""
    echo "Options:"
    echo "  --help, -h  Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Run with defaults"
    echo "  BASE_URL=http://staging.example.com $0  # Test staging environment"
    echo ""
    exit 0
fi

# Run main function
main "$@"