import http from 'k6/http';
import { check, sleep } from 'k6';

// Smoke test configuration - minimal load to verify basic functionality
export const options = {
  vus: 1, // 1 virtual user
  duration: '1m', // Run for 1 minute
  thresholds: {
    http_req_duration: ['p(95)<1000'], // 95% of requests under 1s
    http_req_failed: ['rate<0.01'],    // Less than 1% failures
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const API_TOKEN = __ENV.API_TOKEN || 'test-token';

export function setup() {
  console.log(`Running smoke test against ${BASE_URL}`);
  return { baseUrl: BASE_URL, token: API_TOKEN };
}

export default function (data) {
  const params = {
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${data.token}`,
    },
  };

  // Test 1: Health check
  let response = http.get(`${data.baseUrl}/health`);
  check(response, {
    'Health check status is 200': (r) => r.status === 200,
    'Health check has valid response': (r) => {
      try {
        const json = r.json();
        return json.status === 'healthy' || json.status === 'ok';
      } catch (e) {
        return false;
      }
    },
  });

  sleep(1);

  // Test 2: Basic Q&A functionality
  const qaPayload = JSON.stringify({
    query: 'What is the ticker symbol for Apple?',
    ticker: 'AAPL',
    max_results: 1
  });

  response = http.post(`${data.baseUrl}/api/qa`, qaPayload, params);
  check(response, {
    'Q&A endpoint accessible': (r) => r.status === 200 || r.status === 400,
    'Q&A response is JSON': (r) => {
      try {
        r.json();
        return true;
      } catch (e) {
        return false;
      }
    },
  });

  sleep(2);

  // Test 3: Metrics endpoint
  response = http.get(`${data.baseUrl}/metrics`);
  check(response, {
    'Metrics endpoint accessible': (r) => r.status === 200,
  });

  sleep(1);

  // Test 4: API documentation (if available)
  response = http.get(`${data.baseUrl}/docs`);
  check(response, {
    'Docs endpoint accessible': (r) => r.status === 200 || r.status === 404,
  });

  sleep(2);
}

export function teardown(data) {
  console.log('Smoke test completed - basic functionality verified');
}