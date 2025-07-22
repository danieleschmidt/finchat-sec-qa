"""Tests for the FinChat SEC QA SDK client."""
from __future__ import annotations

import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any, Optional
import httpx


class TestFinChatSDKClient:
    """Test the SDK client for external developer consumption."""
    
    def test_sdk_client_can_be_imported(self):
        """Test that SDK client can be imported from the main package."""
        try:
            from finchat_sec_qa.sdk import FinChatClient
            assert FinChatClient is not None
        except ImportError:
            pytest.fail("FinChatClient should be importable from finchat_sec_qa.sdk")
    
    def test_sync_client_initialization(self):
        """Test synchronous client initialization with proper configuration."""
        from finchat_sec_qa.sdk import FinChatClient
        
        # Test default initialization
        client = FinChatClient()
        assert client.base_url == "http://localhost:8000"
        assert client.timeout == 30
        
        # Test custom initialization
        client = FinChatClient(
            base_url="https://api.finchat.example.com",
            timeout=60,
            api_key="test-key"
        )
        assert client.base_url == "https://api.finchat.example.com"
        assert client.timeout == 60
        assert client.api_key == "test-key"
    
    def test_async_client_initialization(self):
        """Test asynchronous client initialization."""
        from finchat_sec_qa.sdk import AsyncFinChatClient
        
        # Test default initialization
        client = AsyncFinChatClient()
        assert client.base_url == "http://localhost:8000"
        assert client.timeout == 30
        
        # Test custom initialization
        client = AsyncFinChatClient(
            base_url="https://api.finchat.example.com",
            timeout=60,
            api_key="test-key"
        )
        assert client.base_url == "https://api.finchat.example.com"
        assert client.timeout == 60
        assert client.api_key == "test-key"
    
    def test_sync_client_query_method(self):
        """Test synchronous client query method with proper types."""
        from finchat_sec_qa.sdk import FinChatClient
        
        client = FinChatClient(base_url="https://test.example.com")
        
        # Mock the HTTP response
        with patch('httpx.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "answer": "Apple Inc. reported revenue of $394.3 billion in fiscal 2022.",
                "citations": [
                    {
                        "text": "Revenue was $394.3 billion",
                        "source": "AAPL 10-K 2022",
                        "page": 1,
                        "start_pos": 1500,
                        "end_pos": 1525
                    }
                ]
            }
            mock_post.return_value = mock_response
            
            # Test query method
            result = client.query(
                question="What was Apple's revenue in 2022?",
                ticker="AAPL",
                form_type="10-K",
                limit=1
            )
            
            # Verify the request was made correctly
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[0][0] == "https://test.example.com/query"
            
            # Verify response structure
            assert "answer" in result
            assert "citations" in result
            assert result["answer"] == "Apple Inc. reported revenue of $394.3 billion in fiscal 2022."
            assert len(result["citations"]) == 1
            assert result["citations"][0]["source"] == "AAPL 10-K 2022"
    
    @pytest.mark.asyncio
    async def test_async_client_query_method(self):
        """Test asynchronous client query method."""
        from finchat_sec_qa.sdk import AsyncFinChatClient
        
        client = AsyncFinChatClient(base_url="https://test.example.com")
        
        # Mock the async HTTP response
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "answer": "Apple Inc. reported revenue of $394.3 billion in fiscal 2022.",
                "citations": []
            }
            mock_post.return_value = mock_response
            
            # Test async query method
            result = await client.query(
                question="What was Apple's revenue in 2022?",
                ticker="AAPL",
                form_type="10-K"
            )
            
            # Verify response
            assert "answer" in result
            assert result["answer"] == "Apple Inc. reported revenue of $394.3 billion in fiscal 2022."
    
    def test_sync_client_risk_analysis(self):
        """Test synchronous client risk analysis method."""
        from finchat_sec_qa.sdk import FinChatClient
        
        client = FinChatClient(base_url="https://test.example.com")
        
        with patch('httpx.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "sentiment": "positive",
                "flags": ["growth", "expansion"]
            }
            mock_post.return_value = mock_response
            
            result = client.analyze_risk("The company shows strong financial performance and growth prospects.")
            
            assert result["sentiment"] == "positive"
            assert "growth" in result["flags"]
    
    def test_client_error_handling(self):
        """Test client error handling for various HTTP error cases."""
        from finchat_sec_qa.sdk import FinChatClient, FinChatAPIError, FinChatNotFoundError, FinChatValidationError
        
        client = FinChatClient(base_url="https://test.example.com")
        
        # Test 400 error (validation error)
        with patch('httpx.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 400
            mock_response.json.return_value = {"detail": "Invalid ticker symbol"}
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "400 Bad Request", request=Mock(), response=mock_response
            )
            mock_post.return_value = mock_response
            
            with pytest.raises(FinChatValidationError) as exc_info:
                client.query("What is revenue?", "INVALID", "10-K")
            
            assert "Invalid ticker symbol" in str(exc_info.value)
        
        # Test 404 error (not found)
        with patch('httpx.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 404
            mock_response.json.return_value = {"detail": "Filing not found"}
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "404 Not Found", request=Mock(), response=mock_response
            )
            mock_post.return_value = mock_response
            
            with pytest.raises(FinChatNotFoundError) as exc_info:
                client.query("What is revenue?", "AAPL", "10-K")
            
            assert "Filing not found" in str(exc_info.value)
        
        # Test 500 error (generic API error)
        with patch('httpx.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.json.return_value = {"detail": "Internal server error"}
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "500 Internal Server Error", request=Mock(), response=mock_response
            )
            mock_post.return_value = mock_response
            
            with pytest.raises(FinChatAPIError) as exc_info:
                client.query("What is revenue?", "AAPL", "10-K")
            
            assert "Internal server error" in str(exc_info.value)
    
    def test_client_health_check(self):
        """Test client health check method."""
        from finchat_sec_qa.sdk import FinChatClient
        
        client = FinChatClient(base_url="https://test.example.com")
        
        with patch('httpx.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "status": "healthy",
                "version": "1.4.5",
                "services": {
                    "edgar_client": "ready",
                    "qa_engine": "ready",
                    "query_handler": "ready"
                }
            }
            mock_get.return_value = mock_response
            
            result = client.health_check()
            
            assert result["status"] == "healthy"
            assert result["version"] == "1.4.5"
            assert result["services"]["qa_engine"] == "ready"
    
    def test_client_context_manager_support(self):
        """Test that clients support context manager protocol."""
        from finchat_sec_qa.sdk import FinChatClient, AsyncFinChatClient
        
        # Test sync client
        with FinChatClient() as client:
            assert client is not None
            assert hasattr(client, 'query')
        
        # Test async client - just verify it has the required methods
        async_client = AsyncFinChatClient()
        assert hasattr(async_client, '__aenter__')
        assert hasattr(async_client, '__aexit__')
    
    def test_client_authentication_header(self):
        """Test that client properly sets authentication headers when api_key is provided."""
        from finchat_sec_qa.sdk import FinChatClient
        
        client = FinChatClient(api_key="test-api-key-123")
        
        with patch('httpx.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"answer": "Test answer", "citations": []}
            mock_post.return_value = mock_response
            
            client.query("Test question", "AAPL", "10-K")
            
            # Verify authentication header was set
            call_kwargs = mock_post.call_args[1]
            assert "headers" in call_kwargs
            assert call_kwargs["headers"]["Authorization"] == "Bearer test-api-key-123"
    
    def test_sdk_type_annotations(self):
        """Test that SDK classes have proper type annotations."""
        from finchat_sec_qa.sdk import FinChatClient, AsyncFinChatClient
        import inspect
        
        # Check sync client method signatures
        query_sig = inspect.signature(FinChatClient.query)
        assert 'question' in query_sig.parameters
        assert 'ticker' in query_sig.parameters
        assert 'form_type' in query_sig.parameters
        assert 'limit' in query_sig.parameters
        
        # Check async client method signatures  
        async_query_sig = inspect.signature(AsyncFinChatClient.query)
        assert 'question' in async_query_sig.parameters
        assert 'ticker' in async_query_sig.parameters


class TestSDKDataModels:
    """Test SDK data models and type definitions."""
    
    def test_query_response_model(self):
        """Test QueryResponse model structure and validation."""
        from finchat_sec_qa.sdk import QueryResponse, Citation
        
        # Test valid response creation
        citations = [
            Citation(
                text="Revenue was $394.3 billion",
                source="AAPL 10-K 2022",
                page=1,
                start_pos=1500,
                end_pos=1525
            )
        ]
        
        response = QueryResponse(
            answer="Apple Inc. reported revenue of $394.3 billion in fiscal 2022.",
            citations=citations
        )
        
        assert response.answer == "Apple Inc. reported revenue of $394.3 billion in fiscal 2022."
        assert len(response.citations) == 1
        assert response.citations[0].source == "AAPL 10-K 2022"
    
    def test_risk_analysis_response_model(self):
        """Test RiskAnalysisResponse model."""
        from finchat_sec_qa.sdk import RiskAnalysisResponse
        
        response = RiskAnalysisResponse(
            sentiment="positive",
            flags=["growth", "expansion"]
        )
        
        assert response.sentiment == "positive"
        assert response.flags == ["growth", "expansion"]
    
    def test_health_check_response_model(self):
        """Test HealthCheckResponse model."""
        from finchat_sec_qa.sdk import HealthCheckResponse
        
        response = HealthCheckResponse(
            status="healthy",
            version="1.4.5",
            timestamp="2024-01-01T12:00:00Z",
            services={
                "edgar_client": "ready",
                "qa_engine": "ready"
            }
        )
        
        assert response.status == "healthy"
        assert response.version == "1.4.5"
        assert response.services["edgar_client"] == "ready"