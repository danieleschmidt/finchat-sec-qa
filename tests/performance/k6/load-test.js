import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const responseTime = new Trend('response_time');

// Test configuration
export const options = {
  stages: [
    { duration: '1m', target: 10 },   // Ramp up to 10 users
    { duration: '3m', target: 50 },   // Stay at 50 users for 3 minutes
    { duration: '1m', target: 100 },  // Ramp up to 100 users
    { duration: '2m', target: 100 },  // Stay at 100 users for 2 minutes
    { duration: '1m', target: 0 },    // Ramp down to 0 users
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'],  // 95% of requests must complete below 500ms
    http_req_failed: ['rate<0.05'],    // Error rate must be less than 5%
    errors: ['rate<0.05'],             // Custom error rate threshold
  },
};

// Base URL configuration
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const API_TOKEN = __ENV.API_TOKEN || 'test-token';

// Test data
const testQueries = [
  'What is the revenue of Apple?',
  'Tell me about Tesla financial performance',
  'What are Microsoft main business segments?',
  'Show me Amazon quarterly earnings',
  'What is Google advertising revenue?'
];

const testTickers = ['AAPL', 'TSLA', 'MSFT', 'AMZN', 'GOOGL'];

export function setup() {
  // Health check before starting the test
  const healthCheck = http.get(`${BASE_URL}/health`);
  check(healthCheck, {
    'Health check passed': (r) => r.status === 200,
  });
  
  console.log(`Starting load test against ${BASE_URL}`);
  return { baseUrl: BASE_URL, token: API_TOKEN };
}

export default function (data) {
  const params = {
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${data.token}`,
    },
  };

  // Test 1: Health check endpoint
  let response = http.get(`${data.baseUrl}/health`, params);
  check(response, {
    'Health check status is 200': (r) => r.status === 200,
    'Health check response time < 100ms': (r) => r.timings.duration < 100,
  });
  errorRate.add(response.status !== 200);
  responseTime.add(response.timings.duration);

  sleep(1);

  // Test 2: Metrics endpoint
  response = http.get(`${data.baseUrl}/metrics`, params);
  check(response, {
    'Metrics endpoint status is 200': (r) => r.status === 200,
    'Metrics response time < 200ms': (r) => r.timings.duration < 200,
  });
  errorRate.add(response.status !== 200);
  responseTime.add(response.timings.duration);

  sleep(1);

  // Test 3: Q&A endpoint with random query
  const randomQuery = testQueries[Math.floor(Math.random() * testQueries.length)];
  const randomTicker = testTickers[Math.floor(Math.random() * testTickers.length)];
  
  const qaPayload = JSON.stringify({
    query: randomQuery,
    ticker: randomTicker,
    max_results: 5
  });

  response = http.post(`${data.baseUrl}/api/qa`, qaPayload, params);
  check(response, {
    'Q&A endpoint status is 200': (r) => r.status === 200,
    'Q&A response time < 2000ms': (r) => r.timings.duration < 2000,
    'Q&A response has data': (r) => {
      try {
        const json = r.json();
        return json && (json.answer || json.results);
      } catch (e) {
        return false;
      }
    },
  });
  errorRate.add(response.status !== 200);
  responseTime.add(response.timings.duration);

  sleep(2);

  // Test 4: Company analysis endpoint
  response = http.get(`${data.baseUrl}/api/company/${randomTicker}/analysis`, params);
  check(response, {
    'Company analysis status is 200 or 404': (r) => r.status === 200 || r.status === 404,
    'Company analysis response time < 1500ms': (r) => r.timings.duration < 1500,
  });
  errorRate.add(response.status !== 200 && response.status !== 404);
  responseTime.add(response.timings.duration);

  sleep(1);
}

export function teardown(data) {
  console.log('Load test completed');
  
  // Final health check
  const healthCheck = http.get(`${data.baseUrl}/health`);
  check(healthCheck, {
    'Final health check passed': (r) => r.status === 200,
  });
}