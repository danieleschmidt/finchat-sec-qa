#!/usr/bin/env python3
"""
Terragon SDLC v4.0 Integration Test Suite
Comprehensive validation of autonomous execution capabilities
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from finchat_sec_qa.autonomous_execution_engine import (
    create_autonomous_execution_engine,
    ExecutionStrategy
)
from finchat_sec_qa.advanced_error_recovery import get_error_recovery
from finchat_sec_qa.enhanced_security_hardening import get_security_hardening
from finchat_sec_qa.quantum_hyperscale_orchestrator import get_hyperscale_orchestrator
from finchat_sec_qa.ml_performance_accelerator import get_performance_accelerator


class TerrragonSDLCIntegrationTester:
    """Comprehensive integration tester for Terragon SDLC v4.0"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.test_results = []
        self.start_time = time.time()
    
    async def run_comprehensive_tests(self):
        """Run all integration tests"""
        print("🚀 Starting Terragon SDLC v4.0 Comprehensive Integration Tests")
        print("=" * 70)
        
        # Test categories
        test_suites = [
            ("Autonomous Execution Engine", self.test_autonomous_execution),
            ("Advanced Error Recovery", self.test_error_recovery),
            ("Enhanced Security Hardening", self.test_security_hardening),
            ("Quantum Hyperscale Orchestrator", self.test_hyperscale_orchestrator),
            ("ML Performance Accelerator", self.test_performance_accelerator),
            ("End-to-End Integration", self.test_end_to_end_integration)
        ]
        
        for suite_name, test_func in test_suites:
            print(f"\n🔧 Testing: {suite_name}")
            print("-" * 50)
            
            try:
                start_time = time.time()
                results = await test_func()
                execution_time = time.time() - start_time
                
                passed = sum(1 for r in results if r["passed"])
                total = len(results)
                
                print(f"✅ {suite_name}: {passed}/{total} tests passed ({execution_time:.2f}s)")
                
                self.test_results.extend(results)
                
            except Exception as e:
                print(f"❌ {suite_name}: Failed with error: {e}")
                self.test_results.append({
                    "suite": suite_name,
                    "test": "suite_execution",
                    "passed": False,
                    "error": str(e)
                })
        
        # Generate final report
        self.generate_final_report()
    
    async def test_autonomous_execution(self):
        """Test autonomous execution engine"""
        results = []
        
        try:
            # Test 1: Engine Creation
            engine = await create_autonomous_execution_engine()
            results.append({
                "suite": "Autonomous Execution",
                "test": "engine_creation",
                "passed": engine is not None,
                "details": "Engine created successfully"
            })
            
            # Test 2: Context Validation
            context_valid = (
                hasattr(engine, 'context') and
                engine.context.project_type == "financial_ai"
            )
            results.append({
                "suite": "Autonomous Execution",
                "test": "context_validation",
                "passed": context_valid,
                "details": f"Context: {engine.context.project_type if hasattr(engine, 'context') else 'None'}"
            })
            
            # Test 3: Phase Handler Initialization
            handlers_initialized = (
                hasattr(engine, 'phase_handlers') and
                len(engine.phase_handlers) > 0
            )
            results.append({
                "suite": "Autonomous Execution",
                "test": "phase_handlers",
                "passed": handlers_initialized,
                "details": f"Handlers: {len(engine.phase_handlers) if hasattr(engine, 'phase_handlers') else 0}"
            })
            
            # Test 4: Decision Weights
            weights_valid = (
                hasattr(engine, 'decision_weights') and
                abs(sum(engine.decision_weights.values()) - 1.0) < 0.01
            )
            results.append({
                "suite": "Autonomous Execution",
                "test": "decision_weights",
                "passed": weights_valid,
                "details": f"Weight sum: {sum(engine.decision_weights.values()) if hasattr(engine, 'decision_weights') else 'N/A'}"
            })
            
        except Exception as e:
            results.append({
                "suite": "Autonomous Execution",
                "test": "execution_error",
                "passed": False,
                "error": str(e)
            })
        
        return results
    
    async def test_error_recovery(self):
        """Test advanced error recovery system"""
        results = []
        
        try:
            recovery_system = get_error_recovery()
            
            # Test 1: System Initialization
            results.append({
                "suite": "Error Recovery",
                "test": "system_initialization",
                "passed": recovery_system is not None,
                "details": "Recovery system initialized"
            })
            
            # Test 2: Strategy Configuration
            strategies_configured = (
                hasattr(recovery_system, 'recovery_strategies') and
                len(recovery_system.recovery_strategies) > 0
            )
            results.append({
                "suite": "Error Recovery",
                "test": "strategies_configured",
                "passed": strategies_configured,
                "details": f"Strategies: {len(recovery_system.recovery_strategies) if hasattr(recovery_system, 'recovery_strategies') else 0}"
            })
            
            # Test 3: Simulate Error Handling
            test_error = ValueError("Test error for recovery")
            recovery_result = await recovery_system.handle_error(
                test_error, 
                {"test": True}, 
                "test_component", 
                "test_operation"
            )
            
            results.append({
                "suite": "Error Recovery",
                "test": "error_handling",
                "passed": recovery_result is not None,
                "details": f"Recovery attempted: {recovery_result.strategy_used if recovery_result else 'None'}"
            })
            
            # Test 4: Statistics Collection
            stats = recovery_system.get_recovery_statistics()
            results.append({
                "suite": "Error Recovery",
                "test": "statistics_collection",
                "passed": isinstance(stats, dict) and len(stats) > 0,
                "details": f"Stats keys: {list(stats.keys()) if isinstance(stats, dict) else 'Invalid'}"
            })
            
        except Exception as e:
            results.append({
                "suite": "Error Recovery",
                "test": "execution_error",
                "passed": False,
                "error": str(e)
            })
        
        return results
    
    async def test_security_hardening(self):
        """Test enhanced security hardening system"""
        results = []
        
        try:
            security_system = get_security_hardening()
            
            # Test 1: System Initialization
            results.append({
                "suite": "Security Hardening",
                "test": "system_initialization",
                "passed": security_system is not None,
                "details": "Security system initialized"
            })
            
            # Test 2: Encryption Capabilities
            test_data = b"Test data for encryption"
            try:
                encrypted = await security_system.encrypt_data(test_data)
                decrypted = await security_system.decrypt_data(encrypted)
                encryption_works = decrypted == test_data
            except Exception:
                encryption_works = False
            
            results.append({
                "suite": "Security Hardening",
                "test": "encryption_decryption",
                "passed": encryption_works,
                "details": "Encryption/decryption cycle tested"
            })
            
            # Test 3: Threat Scanning
            threat_assessment = await security_system.scan_for_threats(
                "SELECT * FROM users WHERE id = 1", 
                {"source": "test"}
            )
            
            results.append({
                "suite": "Security Hardening",
                "test": "threat_scanning",
                "passed": threat_assessment is not None,
                "details": f"Threat level: {threat_assessment.threat_level.value if threat_assessment else 'None'}"
            })
            
            # Test 4: Security Report Generation
            report = await security_system.generate_security_report()
            results.append({
                "suite": "Security Hardening",
                "test": "security_reporting",
                "passed": isinstance(report, dict) and "summary" in report,
                "details": f"Report sections: {list(report.keys()) if isinstance(report, dict) else 'Invalid'}"
            })
            
        except Exception as e:
            results.append({
                "suite": "Security Hardening",
                "test": "execution_error",
                "passed": False,
                "error": str(e)
            })
        
        return results
    
    async def test_hyperscale_orchestrator(self):
        """Test quantum hyperscale orchestrator"""
        results = []
        
        try:
            orchestrator = get_hyperscale_orchestrator()
            
            # Test 1: System Initialization
            results.append({
                "suite": "Hyperscale Orchestrator",
                "test": "system_initialization",
                "passed": orchestrator is not None,
                "details": "Orchestrator initialized"
            })
            
            # Test 2: Resource Pool Configuration
            pools_configured = (
                hasattr(orchestrator, 'resource_pools') and
                len(orchestrator.resource_pools) > 0
            )
            results.append({
                "suite": "Hyperscale Orchestrator",
                "test": "resource_pools",
                "passed": pools_configured,
                "details": f"Pools: {list(orchestrator.resource_pools.keys()) if hasattr(orchestrator, 'resource_pools') else 'None'}"
            })
            
            # Test 3: Scaling Orchestration
            scaling_decisions = await orchestrator.orchestrate_scaling()
            results.append({
                "suite": "Hyperscale Orchestrator",
                "test": "scaling_orchestration",
                "passed": isinstance(scaling_decisions, list),
                "details": f"Decisions generated: {len(scaling_decisions)}"
            })
            
            # Test 4: Status Reporting
            status = orchestrator.get_orchestrator_status()
            results.append({
                "suite": "Hyperscale Orchestrator",
                "test": "status_reporting",
                "passed": isinstance(status, dict) and "current_capacity" in status,
                "details": f"Status keys: {list(status.keys()) if isinstance(status, dict) else 'Invalid'}"
            })
            
        except Exception as e:
            results.append({
                "suite": "Hyperscale Orchestrator",
                "test": "execution_error",
                "passed": False,
                "error": str(e)
            })
        
        return results
    
    async def test_performance_accelerator(self):
        """Test ML performance accelerator"""
        results = []
        
        try:
            accelerator = get_performance_accelerator()
            
            # Test 1: System Initialization
            results.append({
                "suite": "Performance Accelerator",
                "test": "system_initialization",
                "passed": accelerator is not None,
                "details": "Accelerator initialized"
            })
            
            # Test 2: ML Models Initialization
            models_ready = (
                hasattr(accelerator, 'performance_predictor') and
                hasattr(accelerator, 'anomaly_detector')
            )
            results.append({
                "suite": "Performance Accelerator",
                "test": "ml_models",
                "passed": models_ready,
                "details": "ML models initialized"
            })
            
            # Test 3: Performance Profiling
            def test_function(x):
                return x * 2 + 1
            
            result, profile = await accelerator.profile_operation(
                "test_operation",
                "mathematical",
                test_function,
                5
            )
            
            profiling_works = result == 11 and profile is not None
            results.append({
                "suite": "Performance Accelerator",
                "test": "performance_profiling",
                "passed": profiling_works,
                "details": f"Result: {result}, Profile: {profile.operation_id if profile else 'None'}"
            })
            
            # Test 4: Status Reporting
            status = accelerator.get_accelerator_status()
            results.append({
                "suite": "Performance Accelerator",
                "test": "status_reporting",
                "passed": isinstance(status, dict) and "performance_profiles" in status,
                "details": f"Profiles: {status.get('performance_profiles', 0)}"
            })
            
        except Exception as e:
            results.append({
                "suite": "Performance Accelerator",
                "test": "execution_error",
                "passed": False,
                "error": str(e)
            })
        
        return results
    
    async def test_end_to_end_integration(self):
        """Test end-to-end system integration"""
        results = []
        
        try:
            # Test 1: All Systems Available
            systems_available = all([
                get_error_recovery() is not None,
                get_security_hardening() is not None,
                get_hyperscale_orchestrator() is not None,
                get_performance_accelerator() is not None
            ])
            
            results.append({
                "suite": "End-to-End Integration",
                "test": "all_systems_available",
                "passed": systems_available,
                "details": "All core systems initialized"
            })
            
            # Test 2: Cross-System Communication
            # Simulate a workflow that uses multiple systems
            security_system = get_security_hardening()
            orchestrator = get_hyperscale_orchestrator()
            
            # Security scan
            threat_result = await security_system.scan_for_threats("test input", {"test": True})
            
            # Scaling orchestration
            scaling_result = await orchestrator.orchestrate_scaling()
            
            communication_works = (
                threat_result is not None and
                isinstance(scaling_result, list)
            )
            
            results.append({
                "suite": "End-to-End Integration",
                "test": "cross_system_communication",
                "passed": communication_works,
                "details": f"Security scan: {threat_result.threat_level.value if threat_result else 'None'}, Scaling decisions: {len(scaling_result)}"
            })
            
            # Test 3: Performance Under Load
            async def simulate_load():
                tasks = []
                for i in range(10):
                    task = asyncio.create_task(
                        get_performance_accelerator().profile_operation(
                            f"load_test_{i}",
                            "load_test",
                            lambda x: x ** 2,
                            i
                        )
                    )
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                return len([r for r in results if not isinstance(r, Exception)])
            
            successful_operations = await simulate_load()
            load_test_passed = successful_operations >= 8  # 80% success rate
            
            results.append({
                "suite": "End-to-End Integration",
                "test": "performance_under_load",
                "passed": load_test_passed,
                "details": f"Successful operations: {successful_operations}/10"
            })
            
            # Test 4: System Resilience
            # Test error recovery integration
            error_recovery = get_error_recovery()
            test_error = RuntimeError("Integration test error")
            
            recovery_result = await error_recovery.handle_error(
                test_error,
                {"integration_test": True},
                "integration_component",
                "resilience_test"
            )
            
            resilience_passed = recovery_result is not None
            results.append({
                "suite": "End-to-End Integration",
                "test": "system_resilience",
                "passed": resilience_passed,
                "details": f"Recovery strategy: {recovery_result.strategy_used.value if recovery_result else 'None'}"
            })
            
        except Exception as e:
            results.append({
                "suite": "End-to-End Integration",
                "test": "execution_error",
                "passed": False,
                "error": str(e)
            })
        
        return results
    
    def generate_final_report(self):
        """Generate comprehensive test report"""
        total_execution_time = time.time() - self.start_time
        
        # Calculate statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.get("passed", False))
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Group by suite
        suites = {}
        for result in self.test_results:
            suite = result.get("suite", "Unknown")
            if suite not in suites:
                suites[suite] = {"total": 0, "passed": 0, "failed": 0}
            
            suites[suite]["total"] += 1
            if result.get("passed", False):
                suites[suite]["passed"] += 1
            else:
                suites[suite]["failed"] += 1
        
        # Print final report
        print("\n" + "=" * 70)
        print("🎯 TERRAGON SDLC v4.0 INTEGRATION TEST REPORT")
        print("=" * 70)
        
        print(f"\n📊 Overall Results:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {passed_tests} ✅")
        print(f"   Failed: {failed_tests} ❌")
        print(f"   Success Rate: {success_rate:.1f}%")
        print(f"   Execution Time: {total_execution_time:.2f}s")
        
        print(f"\n📋 Results by Suite:")
        for suite_name, stats in suites.items():
            success_rate = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
            status_emoji = "✅" if success_rate == 100 else "⚠️" if success_rate >= 80 else "❌"
            print(f"   {status_emoji} {suite_name}: {stats['passed']}/{stats['total']} ({success_rate:.1f}%)")
        
        # Show failed tests
        failed_results = [r for r in self.test_results if not r.get("passed", False)]
        if failed_results:
            print(f"\n❌ Failed Tests:")
            for result in failed_results:
                suite = result.get("suite", "Unknown")
                test = result.get("test", "Unknown")
                error = result.get("error", result.get("details", "No details"))
                print(f"   - {suite}/{test}: {error}")
        
        # Save detailed report
        report_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "execution_time": total_execution_time,
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": success_rate
            },
            "suites": suites,
            "detailed_results": self.test_results
        }
        
        report_file = Path("terragon_sdlc_integration_report.json")
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        
        # Final verdict
        if success_rate >= 95:
            print("\n🎉 TERRAGON SDLC v4.0 INTEGRATION: EXCELLENT!")
            print("   System is ready for production deployment.")
        elif success_rate >= 85:
            print("\n✅ TERRAGON SDLC v4.0 INTEGRATION: GOOD")
            print("   System is functional with minor issues.")
        elif success_rate >= 70:
            print("\n⚠️ TERRAGON SDLC v4.0 INTEGRATION: NEEDS ATTENTION")
            print("   System has significant issues requiring resolution.")
        else:
            print("\n❌ TERRAGON SDLC v4.0 INTEGRATION: CRITICAL ISSUES")
            print("   System requires major fixes before deployment.")


async def main():
    """Run the integration test suite"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    tester = TerrragonSDLCIntegrationTester()
    await tester.run_comprehensive_tests()


if __name__ == "__main__":
    asyncio.run(main())