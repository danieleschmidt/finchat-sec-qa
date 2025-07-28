import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const responseTime = new Trend('response_time');
const requestsPerSecond = new Counter('requests_per_second');

// Stress test configuration - pushing system beyond normal capacity
export const options = {
  stages: [
    { duration: '1m', target: 50 },   // Warm up
    { duration: '2m', target: 200 },  // Ramp up to 200 users
    { duration: '3m', target: 500 },  // Spike to 500 users
    { duration: '2m', target: 1000 }, // Stress spike to 1000 users
    { duration: '3m', target: 500 },  // Scale back to 500
    { duration: '2m', target: 200 },  // Scale down to 200
    { duration: '1m', target: 0 },    // Cool down
  ],
  thresholds: {
    http_req_duration: ['p(95)<2000'],  // 95% of requests under 2s (relaxed for stress)
    http_req_failed: ['rate<0.1'],     // Allow up to 10% failure rate under stress
    errors: ['rate<0.1'],              // Custom error threshold
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const API_TOKEN = __ENV.API_TOKEN || 'test-token';

// Stress test data - more intensive queries
const stressQueries = [
  'Provide detailed financial analysis for Apple including revenue, profit margins, and growth trends',
  'Compare Tesla financial performance with traditional automotive companies',
  'What are the key risk factors mentioned in Microsoft latest 10-K filing?',
  'Analyze Amazon AWS revenue growth and market share trends',
  'What are Google main sources of revenue and how have they changed over time?'
];

const allTickers = ['AAPL', 'TSLA', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'NFLX', 'CRM', 'ORCL'];

export function setup() {
  console.log(`Starting stress test against ${BASE_URL}`);
  console.log('This test will push the system beyond normal operating capacity');
  
  // Verify system is healthy before stress testing
  const healthCheck = http.get(`${BASE_URL}/health`);
  if (healthCheck.status !== 200) {
    throw new Error('System is not healthy - aborting stress test');
  }
  
  return { baseUrl: BASE_URL, token: API_TOKEN };
}

export default function (data) {
  const params = {
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${data.token}`,
    },
    timeout: '10s', // Longer timeout for stress conditions
  };

  requestsPerSecond.add(1);

  // Heavy Q&A request
  const randomQuery = stressQueries[Math.floor(Math.random() * stressQueries.length)];
  const randomTicker = allTickers[Math.floor(Math.random() * allTickers.length)];
  
  const qaPayload = JSON.stringify({
    query: randomQuery,
    ticker: randomTicker,
    max_results: 10, // Request more results to increase load
    include_context: true
  });

  let response = http.post(`${data.baseUrl}/api/qa`, qaPayload, params);
  check(response, {
    'Q&A request completed': (r) => r.status !== 0, // Accept any response, even errors
    'Q&A response time acceptable': (r) => r.timings.duration < 10000, // 10s max
  });
  
  errorRate.add(response.status >= 400 || response.status === 0);
  responseTime.add(response.timings.duration);

  // Minimal sleep to maintain high load
  sleep(0.1);

  // Concurrent company analysis request
  response = http.get(`${data.baseUrl}/api/company/${randomTicker}/analysis?deep=true`, params);
  check(response, {
    'Analysis request completed': (r) => r.status !== 0,
    'Analysis response time acceptable': (r) => r.timings.duration < 8000,
  });
  
  errorRate.add(response.status >= 400 || response.status === 0);
  responseTime.add(response.timings.duration);

  // Very short sleep to maintain stress
  sleep(0.05);
}

export function teardown(data) {
  console.log('Stress test completed');
  
  // Allow system to recover before final check
  sleep(5);
  
  const healthCheck = http.get(`${data.baseUrl}/health`);
  const isHealthy = healthCheck.status === 200;
  
  console.log(`System health after stress test: ${isHealthy ? 'HEALTHY' : 'DEGRADED'}`);
  
  if (!isHealthy) {
    console.warn('System may need time to recover from stress test');
  }
}