import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from .edgar_client import EdgarClient
from .logging_utils import configure_logging
from .qa_engine import FinancialQAEngine
from .risk_intelligence import RiskAnalyzer
from .voice_interface import speak
from .photonic_mlir import FinancialQueryType

logger = logging.getLogger(__name__)


def build_engine(docs: List[Path]) -> FinancialQAEngine:
    engine = FinancialQAEngine()
    for path in docs:
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")
        engine.add_document(path.name, path.read_text())
    return engine


def _cmd_query(args: argparse.Namespace) -> int:
    configure_logging(args.log_level)
    logger.info("Starting query command with %d documents", len(args.documents))
    engine = build_engine([Path(p) for p in args.documents])
    logger.info("Answering question: %s", args.question)
    answer, citations = engine.answer_with_citations(args.question)
    logger.info("Generated answer with %d citations", len(citations))
    print(answer)
    if args.voice:
        logger.debug("Speaking answer aloud")
        speak(answer)
    for c in citations:
        print(f"[{c.doc_id}] {c.text}")
    logger.info("Query command completed successfully")
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    configure_logging(args.log_level)
    logger.info("Starting ingest command for ticker: %s", args.ticker)
    client = EdgarClient("FinChatCLI")
    filings = client.get_recent_filings(
        args.ticker, form_type=args.form_type, limit=args.limit
    )
    logger.info("Found %d filings to download", len(filings))
    for filing in filings:
        logger.debug("Downloading filing: %s", filing.document_url)
        client.download_filing(filing, args.dest)
    logger.info("Ingest command completed successfully")
    return 0


def _cmd_risk(args: argparse.Namespace) -> int:
    configure_logging(args.log_level)
    logger.info("Starting risk analysis for file: %s", args.file)
    analyzer = RiskAnalyzer()
    text = Path(args.file).read_text()
    logger.debug("Loaded text with %d characters", len(text))
    assessment = analyzer.assess(text)
    logger.info("Risk assessment completed with %d flags", len(assessment.flags))
    print(assessment.sentiment)
    for flag in assessment.flags:
        print(flag)
    logger.info("Risk command completed successfully")
    return 0


def _cmd_quantum_query(args: argparse.Namespace) -> int:
    """Quantum-enhanced query processing command."""
    configure_logging(args.log_level)
    logger.info("Starting quantum-enhanced query with %d documents", len(args.documents))
    
    # Build engine with quantum capabilities
    engine = FinancialQAEngine(enable_quantum=True)
    for path in [Path(p) for p in args.documents]:
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")
        engine.add_document(path.name, path.read_text())
    
    logger.info("Processing quantum-enhanced question: %s", args.question)
    
    # Use quantum-enhanced processing
    try:
        result = engine.quantum_enhanced_query(
            question=args.question,
            document_path=args.documents[0],  # Use first document as primary
            quantum_threshold=args.quantum_threshold
        )
        
        # Display results
        result_dict = result.to_dict()
        
        if args.format == "json":
            print(json.dumps(result_dict, indent=2))
        else:
            # Human-readable format
            print("=== QUANTUM-ENHANCED FINANCIAL ANALYSIS ===\n")
            print(f"Query: {args.question}\n")
            print(f"Classical Answer:")
            print(result.classical_answer)
            print(f"\nQuantum-Enhanced Answer:")
            print(result_dict["quantum_enhanced_answer"])
            print(f"\nQuantum Advantage: {result.quantum_result.quantum_advantage:.1f}x")
            print(f"Confidence Score: {result.confidence_score:.2%}")
            print(f"Processing Time: {result_dict['quantum_metadata']['processing_time_ms']:.1f}ms")
            
            if result.citations:
                print(f"\nCitations ({len(result.citations)}):")
                for i, citation in enumerate(result.citations, 1):
                    print(f"  [{i}] {citation.doc_id}: {citation.text[:100]}...")
        
        # Voice output if requested
        if args.voice:
            logger.debug("Speaking quantum-enhanced answer aloud")
            speak(result_dict["quantum_enhanced_answer"])
        
        logger.info("Quantum query command completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Quantum query failed: {e}")
        print(f"Error: {e}")
        return 1


def _cmd_quantum_benchmark(args: argparse.Namespace) -> int:
    """Benchmark quantum vs classical performance."""
    configure_logging(args.log_level)
    logger.info("Starting quantum benchmark")
    
    # Build engine with quantum capabilities
    engine = FinancialQAEngine(enable_quantum=True)
    
    # Load test documents
    test_documents = []
    for path in [Path(p) for p in args.documents]:
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")
        engine.add_document(path.name, path.read_text())
        test_documents.append(str(path))
    
    # Default test queries if none provided
    test_queries = args.queries if hasattr(args, 'queries') and args.queries else [
        "What are the main risk factors?",
        "Analyze the portfolio allocation strategy",
        "What is the expected volatility?",
        "Assess the correlation between assets",
        "Predict future market trends"
    ]
    
    logger.info(f"Benchmarking {len(test_queries)} queries across {len(test_documents)} documents")
    
    try:
        benchmark_results = engine.benchmark_quantum_performance(
            test_queries=test_queries,
            test_documents=test_documents
        )
        
        if args.format == "json":
            print(json.dumps(benchmark_results, indent=2))
        else:
            # Human-readable benchmark report
            print("=== QUANTUM VS CLASSICAL BENCHMARK RESULTS ===\n")
            
            agg = benchmark_results["aggregate_metrics"]
            print(f"Average Quantum Advantage: {agg['average_quantum_advantage']:.1f}x")
            print(f"Average Confidence Improvement: {agg['average_confidence_improvement']:.1%}")
            print(f"Overall Speedup: {agg['overall_speedup']:.1f}x")
            print(f"Total Classical Time: {agg['total_classical_time_ms']:.1f}ms")
            print(f"Total Quantum Time: {agg['total_quantum_time_ms']:.1f}ms")
            
            print(f"\nDetailed Results:")
            for i, result in enumerate(benchmark_results["benchmark_results"], 1):
                print(f"  {i}. {result['query'][:50]}...")
                print(f"     Type: {result['query_type']}")
                print(f"     Quantum Advantage: {result['quantum_advantage']:.1f}x")
                print(f"     Time: {result['classical_time_ms']:.1f}ms → {result['quantum_time_ms']:.1f}ms")
                print()
        
        logger.info("Quantum benchmark completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        print(f"Error: {e}")
        return 1


def _cmd_quantum_capabilities(args: argparse.Namespace) -> int:
    """Display quantum computing capabilities."""
    configure_logging(args.log_level)
    logger.info("Retrieving quantum capabilities")
    
    engine = FinancialQAEngine(enable_quantum=True)
    capabilities = engine.get_quantum_capabilities()
    
    if args.format == "json":
        print(json.dumps(capabilities, indent=2))
    else:
        print("=== PHOTONIC QUANTUM COMPUTING CAPABILITIES ===\n")
        
        if capabilities.get("quantum_enabled", False):
            print("✅ Quantum Processing: ENABLED")
            print(f"📊 Supported Query Types: {', '.join(capabilities['available_query_types'])}")
            print(f"🔬 Maximum Qubits: {capabilities['max_qubits']}")
            print(f"⏱️  Coherence Time: {capabilities['coherence_time_ms']}ms")
            print(f"🎯 Gate Fidelity: {capabilities['gate_fidelity']:.1%}")
            print(f"📈 Quantum Volume: {capabilities['quantum_volume']}")
            print(f"🧮 Supported Algorithms:")
            for algo in capabilities['supported_algorithms']:
                print(f"   • {algo}")
            print(f"🚪 Quantum Gates: {', '.join(capabilities['quantum_gates_supported'])}")
        else:
            print("❌ Quantum Processing: DISABLED")
            print(f"Reason: {capabilities.get('reason', 'unknown')}")
    
    logger.info("Capabilities command completed successfully")
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="FinChat-SEC-QA command line interface"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("query", help="Answer a question from documents")
    q.add_argument("question")
    q.add_argument("documents", nargs="+")
    q.add_argument("-v", "--voice", action="store_true", help="Speak answer")
    q.add_argument("-l", "--log-level", default="INFO", help="Logging level")
    q.set_defaults(func=_cmd_query)

    ing = sub.add_parser("ingest", help="Download filings for a ticker")
    ing.add_argument("ticker")
    ing.add_argument("--form-type", default="10-K")
    ing.add_argument("--limit", type=int, default=1)
    ing.add_argument("--dest", type=Path, default=Path.cwd())
    ing.add_argument("-l", "--log-level", default="INFO", help="Logging level")
    ing.set_defaults(func=_cmd_ingest)

    r = sub.add_parser("risk", help="Assess risk from a text file")
    r.add_argument("file")
    r.add_argument("-l", "--log-level", default="INFO", help="Logging level")
    r.set_defaults(func=_cmd_risk)

    # Quantum-enhanced commands
    qquery = sub.add_parser("quantum-query", help="Quantum-enhanced question answering")
    qquery.add_argument("question", help="Financial question to ask")
    qquery.add_argument("documents", nargs="+", help="Financial documents to analyze")
    qquery.add_argument("-v", "--voice", action="store_true", help="Speak answer")
    qquery.add_argument("-f", "--format", choices=["text", "json"], default="text", help="Output format")
    qquery.add_argument("-t", "--quantum-threshold", type=float, default=0.7, help="Quantum enhancement threshold")
    qquery.add_argument("-l", "--log-level", default="INFO", help="Logging level")
    qquery.set_defaults(func=_cmd_quantum_query)

    qbench = sub.add_parser("quantum-benchmark", help="Benchmark quantum vs classical performance")
    qbench.add_argument("documents", nargs="+", help="Financial documents for benchmarking")
    qbench.add_argument("-q", "--queries", nargs="*", help="Custom test queries")
    qbench.add_argument("-f", "--format", choices=["text", "json"], default="text", help="Output format")
    qbench.add_argument("-l", "--log-level", default="INFO", help="Logging level")
    qbench.set_defaults(func=_cmd_quantum_benchmark)

    qcap = sub.add_parser("quantum-capabilities", help="Display quantum computing capabilities")
    qcap.add_argument("-f", "--format", choices=["text", "json"], default="text", help="Output format")
    qcap.add_argument("-l", "--log-level", default="INFO", help="Logging level")
    qcap.set_defaults(func=_cmd_quantum_capabilities)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
