#!/usr/bin/env python3
"""
Quantum Financial Algorithm Monitoring Demonstration.

This script demonstrates the comprehensive monitoring and observability
features for quantum financial algorithms in production environments.

MONITORING DEMO - Enterprise-Grade Observability
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import logging
import time
import random
import json
from datetime import datetime
import numpy as np

from finchat_sec_qa.quantum_monitoring import (
    QuantumMonitor,
    monitor_quantum_algorithm,
    AlertSeverity,
    MetricType
)

def setup_logging():
    """Setup comprehensive logging for monitoring demonstration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# Simulated quantum algorithm results for demonstration
class SimulatedQuantumResult:
    """Simulated quantum algorithm result with monitoring metrics."""
    
    def __init__(self, quantum_advantage=2.5, fidelity=0.95, circuit_depth=20):
        self.quantum_advantage_score = quantum_advantage
        self.fidelity = fidelity
        self.circuit_depth = circuit_depth

def simulate_quantum_timeseries_analysis():
    """Simulate quantum time series analysis with realistic metrics."""
    # Simulate processing time (100ms to 2s)
    time.sleep(random.uniform(0.1, 2.0))
    
    # Generate realistic quantum metrics with some variability
    quantum_advantage = random.gauss(2.8, 0.4)
    fidelity = random.gauss(0.92, 0.05)
    circuit_depth = random.randint(15, 30)
    
    # Simulate occasional failures (5% failure rate)
    if random.random() < 0.05:
        raise RuntimeError("Quantum circuit execution failed")
    
    return SimulatedQuantumResult(quantum_advantage, fidelity, circuit_depth)

def simulate_quantum_risk_prediction():
    """Simulate quantum risk prediction with monitoring."""
    time.sleep(random.uniform(0.2, 1.5))
    
    quantum_advantage = random.gauss(3.2, 0.5)
    fidelity = random.gauss(0.88, 0.08)
    circuit_depth = random.randint(20, 40)
    
    if random.random() < 0.03:  # 3% failure rate
        raise ValueError("Invalid risk parameters")
    
    return SimulatedQuantumResult(quantum_advantage, fidelity, circuit_depth)

def simulate_quantum_portfolio_optimization():
    """Simulate quantum portfolio optimization with monitoring."""
    time.sleep(random.uniform(0.5, 3.0))  # More complex, longer execution
    
    quantum_advantage = random.gauss(2.5, 0.3)
    fidelity = random.gauss(0.85, 0.1)
    circuit_depth = random.randint(40, 80)
    
    if random.random() < 0.08:  # 8% failure rate (more complex)
        raise RuntimeError("QAOA optimization convergence failed")
    
    return SimulatedQuantumResult(quantum_advantage, fidelity, circuit_depth)

def simulate_photonic_cv_processing():
    """Simulate photonic continuous variable processing."""
    time.sleep(random.uniform(0.05, 0.8))  # Faster photonic processing
    
    quantum_advantage = random.gauss(4.1, 0.6)  # Higher advantage
    fidelity = random.gauss(0.96, 0.03)  # Higher fidelity
    circuit_depth = random.randint(5, 15)  # Shallower circuits
    
    if random.random() < 0.02:  # 2% failure rate (more stable)
        raise RuntimeError("Photonic mode decoherence")
    
    class PhotonicResult(SimulatedQuantumResult):
        def __init__(self, qa, fid, depth):
            super().__init__(qa, fid, depth)
            self.photonic_fidelity = fid  # Add photonic-specific metric
    
    return PhotonicResult(quantum_advantage, fidelity, circuit_depth)

def run_monitoring_demonstration():
    """Run comprehensive monitoring demonstration."""
    logger = logging.getLogger(__name__)
    logger.info("🔍 Starting Quantum Financial Algorithm Monitoring Demo")
    
    # Initialize quantum monitor
    monitor_config = {
        'metrics_buffer_size': 5000,
        'alert_log_file': 'quantum_alerts.log'
    }
    
    quantum_monitor = QuantumMonitor(monitor_config)
    
    # Add custom alert rules
    quantum_monitor.alert_manager.add_alert_rule(
        "quantum_timeseries_execution_time",
        threshold=1000,  # 1 second
        condition="greater_than",
        severity=AlertSeverity.WARNING,
        description="Time series analysis taking too long"
    )
    
    quantum_monitor.alert_manager.add_alert_rule(
        "photonic_cv_processing_quantum_advantage",
        threshold=3.5,
        condition="less_than",
        severity=AlertSeverity.INFO,
        description="Photonic quantum advantage below expected threshold"
    )
    
    # Create monitored algorithm functions
    @monitor_quantum_algorithm(quantum_monitor, "quantum_timeseries")
    def monitored_timeseries_analysis():
        return simulate_quantum_timeseries_analysis()
    
    @monitor_quantum_algorithm(quantum_monitor, "quantum_risk_prediction")
    def monitored_risk_prediction():
        return simulate_quantum_risk_prediction()
    
    @monitor_quantum_algorithm(quantum_monitor, "quantum_portfolio_optimization")
    def monitored_portfolio_optimization():
        return simulate_quantum_portfolio_optimization()
    
    @monitor_quantum_algorithm(quantum_monitor, "photonic_cv_processing")
    def monitored_photonic_processing():
        return simulate_photonic_cv_processing()
    
    # Simulation parameters
    num_iterations = 100
    algorithms = [
        ("Time Series Analysis", monitored_timeseries_analysis),
        ("Risk Prediction", monitored_risk_prediction),
        ("Portfolio Optimization", monitored_portfolio_optimization),
        ("Photonic CV Processing", monitored_photonic_processing)
    ]
    
    print("\\n" + "="*80)
    print("🔍 QUANTUM ALGORITHM MONITORING DEMONSTRATION")
    print("="*80)
    print(f"Running {num_iterations} iterations across {len(algorithms)} algorithms")
    print("Monitoring: Execution time, quantum advantage, fidelity, errors, alerts")
    print("="*80)
    
    # Run simulation
    iteration_results = []
    
    for iteration in range(num_iterations):
        print(f"\\r🔄 Progress: {iteration+1}/{num_iterations} ({(iteration+1)/num_iterations*100:.1f}%)", end="", flush=True)
        
        # Randomly select algorithm to execute
        algorithm_name, algorithm_func = random.choice(algorithms)
        
        iteration_start = time.time()
        success = True
        error_details = None
        
        try:
            result = algorithm_func()
            iteration_results.append({
                'iteration': iteration,
                'algorithm': algorithm_name,
                'success': True,
                'quantum_advantage': result.quantum_advantage_score,
                'fidelity': getattr(result, 'fidelity', getattr(result, 'photonic_fidelity', None)),
                'circuit_depth': result.circuit_depth
            })
            
        except Exception as e:
            success = False
            error_details = str(e)
            iteration_results.append({
                'iteration': iteration,
                'algorithm': algorithm_name,
                'success': False,
                'error': error_details
            })
        
        # Small delay between iterations
        time.sleep(0.1)
    
    print("\\n\\n✅ Simulation completed!")
    
    # Display monitoring results
    print("\\n" + "-"*60)
    print("📊 MONITORING RESULTS")
    print("-"*60)
    
    # Get comprehensive monitoring dashboard
    dashboard = quantum_monitor.get_monitoring_dashboard()
    
    # Display metrics summary
    metrics_summary = dashboard['metrics_summary']
    print(f"\\n📈 METRICS SUMMARY:")
    print(f"   Buffer Size: {metrics_summary['buffer_size']} metrics")
    
    print(f"\\n📊 COUNTERS:")
    for counter_name, count in metrics_summary['counters'].items():
        print(f"   {counter_name}: {count}")
    
    print(f"\\n📏 GAUGES:")
    for gauge_name, value in metrics_summary['gauges'].items():
        print(f"   {gauge_name}: {value:.3f}")
    
    print(f"\\n📈 HISTOGRAM STATISTICS:")
    for hist_name, stats in metrics_summary['histogram_stats'].items():
        if 'execution_time' in hist_name:
            print(f"   {hist_name}:")
            print(f"     Count: {stats['count']}")
            print(f"     Mean: {stats['mean']:.1f}ms")
            print(f"     P95: {stats['p95']:.1f}ms")
            print(f"     P99: {stats['p99']:.1f}ms")
    
    # Display alert summary
    alert_summary = dashboard['alert_summary']
    print(f"\\n🚨 ALERT SUMMARY:")
    print(f"   Total Alerts: {alert_summary['total_alerts']}")
    print(f"   Active Alerts: {alert_summary['active_alerts']}")
    print(f"   Alert Rules: {alert_summary['alert_rules']}")
    
    print(f"\\n🚨 ALERTS BY SEVERITY:")
    for severity, count in alert_summary['alerts_by_severity'].items():
        if count > 0:
            print(f"   {severity.upper()}: {count}")
    
    # Display active alerts
    active_alerts = dashboard['active_alerts']
    if active_alerts:
        print(f"\\n🚨 ACTIVE ALERTS:")
        for alert in active_alerts[:5]:  # Show first 5
            print(f"   [{alert['severity'].upper()}] {alert['title']}")
            print(f"     {alert['description']}")
            print(f"     Time: {alert['timestamp']}")
    
    # Display performance traces
    trace_stats = dashboard['trace_statistics']
    if 'duration_stats' in trace_stats:
        print(f"\\n⏱️  PERFORMANCE TRACES:")
        duration_stats = trace_stats['duration_stats']
        print(f"   Total Traces: {trace_stats['total_traces']}")
        print(f"   Mean Duration: {duration_stats['mean_ms']:.1f}ms")
        print(f"   95th Percentile: {duration_stats['p95_ms']:.1f}ms")
        
        if 'quantum_advantage_stats' in trace_stats:
            qa_stats = trace_stats['quantum_advantage_stats']
            print(f"   Mean Quantum Advantage: {qa_stats['mean']:.2f}x")
            print(f"   QA Range: {qa_stats['min']:.2f}x - {qa_stats['max']:.2f}x")
        
        if 'fidelity_stats' in trace_stats:
            fid_stats = trace_stats['fidelity_stats']
            print(f"   Mean Fidelity: {fid_stats['mean']:.3f}")
            print(f"   Fidelity Range: {fid_stats['min']:.3f} - {fid_stats['max']:.3f}")
    
    # Display operation breakdown
    if 'operations' in trace_stats:
        print(f"\\n🔧 OPERATIONS BREAKDOWN:")
        for operation, count in trace_stats['operations'].items():
            print(f"   {operation}: {count} executions")
    
    # Algorithm-specific analysis
    print("\\n" + "-"*60)
    print("🧪 ALGORITHM PERFORMANCE ANALYSIS")
    print("-"*60)
    
    # Calculate algorithm-specific statistics
    algorithm_stats = {}
    for result in iteration_results:
        alg_name = result['algorithm']
        if alg_name not in algorithm_stats:
            algorithm_stats[alg_name] = {
                'executions': 0,
                'successes': 0,
                'failures': 0,
                'quantum_advantages': [],
                'fidelities': [],
                'circuit_depths': []
            }
        
        stats = algorithm_stats[alg_name]
        stats['executions'] += 1
        
        if result['success']:
            stats['successes'] += 1
            stats['quantum_advantages'].append(result['quantum_advantage'])
            if result['fidelity'] is not None:
                stats['fidelities'].append(result['fidelity'])
            stats['circuit_depths'].append(result['circuit_depth'])
        else:
            stats['failures'] += 1
    
    # Display algorithm statistics
    for alg_name, stats in algorithm_stats.items():
        print(f"\\n🔹 {alg_name.upper()}:")
        
        success_rate = stats['successes'] / stats['executions'] if stats['executions'] > 0 else 0
        print(f"   Executions: {stats['executions']}")
        print(f"   Success Rate: {success_rate:.1%}")
        
        if stats['quantum_advantages']:
            qa_mean = np.mean(stats['quantum_advantages'])
            qa_std = np.std(stats['quantum_advantages'])
            print(f"   Quantum Advantage: {qa_mean:.2f} ± {qa_std:.2f}")
        
        if stats['fidelities']:
            fid_mean = np.mean(stats['fidelities'])
            fid_std = np.std(stats['fidelities'])
            print(f"   Fidelity: {fid_mean:.3f} ± {fid_std:.3f}")
        
        if stats['circuit_depths']:
            depth_mean = np.mean(stats['circuit_depths'])
            depth_std = np.std(stats['circuit_depths'])
            print(f"   Circuit Depth: {depth_mean:.1f} ± {depth_std:.1f}")
        
        if stats['failures'] > 0:
            error_rate = stats['failures'] / stats['executions']
            print(f"   Error Rate: {error_rate:.1%}")
    
    # System health check
    print("\\n" + "-"*60)
    print("🏥 SYSTEM HEALTH CHECK")
    print("-"*60)
    
    health_status = quantum_monitor.health_check()
    print(f"\\n🏥 OVERALL STATUS: {health_status['status'].upper()}")
    
    print(f"\\n🔧 COMPONENT STATUS:")
    for component, status in health_status['components'].items():
        status_emoji = "✅" if status == "healthy" else "⚠️" if "error" in status else "❓"
        print(f"   {status_emoji} {component}: {status}")
    
    if 'statistics' in health_status:
        print(f"\\n📊 HEALTH STATISTICS:")
        for stat_name, stat_value in health_status['statistics'].items():
            print(f"   {stat_name}: {stat_value}")
    
    # Export metrics for external monitoring
    print("\\n" + "-"*60)
    print("📤 METRICS EXPORT")
    print("-"*60)
    
    # Export in Prometheus format
    prometheus_metrics = quantum_monitor.export_metrics("prometheus")
    with open("quantum_metrics.prom", "w") as f:
        f.write(prometheus_metrics)
    print("📤 Prometheus metrics exported to: quantum_metrics.prom")
    
    # Export in JSON format
    json_metrics = quantum_monitor.export_metrics("json")
    with open("quantum_metrics.json", "w") as f:
        f.write(json_metrics)
    print("📤 JSON metrics exported to: quantum_metrics.json")
    
    # Summary and recommendations
    print("\\n" + "="*80)
    print("📋 MONITORING SUMMARY & RECOMMENDATIONS")
    print("="*80)
    
    total_executions = sum(stats['executions'] for stats in algorithm_stats.values())
    total_successes = sum(stats['successes'] for stats in algorithm_stats.values())
    overall_success_rate = total_successes / total_executions if total_executions > 0 else 0
    
    all_qa_values = []
    all_fidelities = []
    
    for stats in algorithm_stats.values():
        all_qa_values.extend(stats['quantum_advantages'])
        all_fidelities.extend(stats['fidelities'])
    
    print(f"\\n📊 OVERALL PERFORMANCE:")
    print(f"   Total Executions: {total_executions}")
    print(f"   Overall Success Rate: {overall_success_rate:.1%}")
    
    if all_qa_values:
        mean_qa = np.mean(all_qa_values)
        print(f"   Average Quantum Advantage: {mean_qa:.2f}x")
    
    if all_fidelities:
        mean_fidelity = np.mean(all_fidelities)
        print(f"   Average Fidelity: {mean_fidelity:.3f}")
    
    print(f"\\n💡 RECOMMENDATIONS:")
    
    if overall_success_rate < 0.95:
        print("   ⚠️  Consider implementing circuit error correction (success rate < 95%)")
    
    if all_qa_values and np.mean(all_qa_values) < 2.0:
        print("   ⚠️  Quantum advantage below 2x - optimize quantum algorithms")
    
    if all_fidelities and np.mean(all_fidelities) < 0.9:
        print("   ⚠️  Low average fidelity - check quantum hardware noise levels")
    
    if alert_summary['active_alerts'] > 0:
        print(f"   🚨 {alert_summary['active_alerts']} active alerts need attention")
    
    print("   ✅ Monitoring system is operational and collecting comprehensive metrics")
    print("   ✅ Performance traces provide detailed quantum execution insights")
    print("   ✅ Alert system is configured for proactive issue detection")
    
    print("\\n🎯 NEXT STEPS:")
    print("   1. Integrate with external monitoring systems (Grafana, Datadog, etc.)")
    print("   2. Set up automated alert notifications (email, Slack, PagerDuty)")
    print("   3. Configure custom alert thresholds based on business requirements")
    print("   4. Implement log aggregation for distributed quantum computing clusters")
    print("   5. Add custom metrics for domain-specific financial performance indicators")
    
    print("\\n✅ MONITORING DEMONSTRATION COMPLETED")
    
    return {
        'total_executions': total_executions,
        'success_rate': overall_success_rate,
        'quantum_advantage': np.mean(all_qa_values) if all_qa_values else 0,
        'fidelity': np.mean(all_fidelities) if all_fidelities else 0,
        'active_alerts': alert_summary['active_alerts'],
        'monitoring_status': health_status['status']
    }

if __name__ == "__main__":
    # Setup logging
    setup_logging()
    
    try:
        results = run_monitoring_demonstration()
        print("\\n🎉 Monitoring demonstration completed successfully!")
        
    except Exception as e:
        logging.error(f"Monitoring demonstration failed: {e}")
        sys.exit(1)