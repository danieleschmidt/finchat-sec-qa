"""Tests for webapp IP address security and X-Forwarded-For validation."""
import pytest
from unittest.mock import Mock, patch
from flask import Flask
from finchat_sec_qa.webapp import _get_client_ip, app


def test_direct_ip_when_no_proxy_headers():
    """Test that direct IP is used when no proxy headers are present."""
    with app.test_request_context('/', environ_base={'REMOTE_ADDR': '192.168.1.100'}):
        ip = _get_client_ip()
        assert ip == '192.168.1.100'


def test_x_forwarded_for_ignored_from_untrusted_source():
    """Test that X-Forwarded-For is ignored when coming from untrusted sources."""
    with app.test_request_context('/', 
                                 headers={'X-Forwarded-For': '10.0.0.1'},
                                 environ_base={'REMOTE_ADDR': '203.0.113.1'}):  # External IP
        ip = _get_client_ip()
        # Should use direct IP since the request doesn't come from a trusted proxy
        assert ip == '203.0.113.1'


def test_x_forwarded_for_trusted_from_private_networks():
    """Test that X-Forwarded-For is trusted when coming from private networks."""
    # Test private IP ranges that are commonly used for load balancers
    trusted_proxy_ips = ['10.0.0.1', '172.16.0.1', '192.168.1.1', '127.0.0.1']
    
    for proxy_ip in trusted_proxy_ips:
        with app.test_request_context('/', 
                                     headers={'X-Forwarded-For': '203.0.113.5'},
                                     environ_base={'REMOTE_ADDR': proxy_ip}):
            ip = _get_client_ip()
            # Should use forwarded IP since request comes from trusted proxy
            assert ip == '203.0.113.5'


def test_x_forwarded_for_multiple_ips_uses_first():
    """Test that first IP is used when X-Forwarded-For has multiple IPs."""
    with app.test_request_context('/', 
                                 headers={'X-Forwarded-For': '203.0.113.5, 10.0.0.1, 192.168.1.1'},
                                 environ_base={'REMOTE_ADDR': '10.0.0.2'}):  # Trusted proxy
        ip = _get_client_ip()
        assert ip == '203.0.113.5'


def test_malformed_x_forwarded_for_header():
    """Test handling of malformed X-Forwarded-For headers."""
    with app.test_request_context('/', 
                                 headers={'X-Forwarded-For': 'not-an-ip'},
                                 environ_base={'REMOTE_ADDR': '10.0.0.1'}):  # Trusted proxy
        ip = _get_client_ip()
        # Should fall back to direct IP when forwarded header is malformed
        assert ip == '10.0.0.1'


def test_empty_x_forwarded_for_header():
    """Test handling of empty X-Forwarded-For headers."""
    with app.test_request_context('/', 
                                 headers={'X-Forwarded-For': ''},
                                 environ_base={'REMOTE_ADDR': '192.168.1.100'}):
        ip = _get_client_ip()
        assert ip == '192.168.1.100'


def test_spoofed_header_from_external_attacker():
    """Test that spoofed headers from external attackers are ignored."""
    # Simulate an attacker trying to spoof their IP
    with app.test_request_context('/', 
                                 headers={'X-Forwarded-For': '127.0.0.1'},  # Trying to spoof localhost
                                 environ_base={'REMOTE_ADDR': '203.0.113.100'}):  # External attacker IP
        ip = _get_client_ip()
        # Should use the actual external IP, not the spoofed one
        assert ip == '203.0.113.100'


def test_x_real_ip_header_support():
    """Test support for X-Real-IP header as secondary option."""
    with app.test_request_context('/', 
                                 headers={'X-Real-IP': '203.0.113.7'},
                                 environ_base={'REMOTE_ADDR': '10.0.0.1'}):  # Trusted proxy
        ip = _get_client_ip()
        # Should use X-Real-IP when from trusted source
        assert ip == '203.0.113.7'


def test_precedence_x_forwarded_for_over_x_real_ip():
    """Test that X-Forwarded-For takes precedence over X-Real-IP."""
    with app.test_request_context('/', 
                                 headers={
                                     'X-Forwarded-For': '203.0.113.8',
                                     'X-Real-IP': '203.0.113.9'
                                 },
                                 environ_base={'REMOTE_ADDR': '10.0.0.1'}):  # Trusted proxy
        ip = _get_client_ip()
        # Should prefer X-Forwarded-For
        assert ip == '203.0.113.8'