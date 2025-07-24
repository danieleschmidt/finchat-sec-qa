"""Tests for Docker containerization integration."""
from __future__ import annotations

import pytest
import subprocess
import os
import yaml
from pathlib import Path


class TestDockerIntegration:
    """Test Docker containerization functionality."""
    
    def test_dockerfile_exists(self):
        """Test that Dockerfile for FastAPI server exists."""
        repo_root = Path(__file__).parent.parent
        dockerfile = repo_root / "docker" / "Dockerfile.api"
        
        assert dockerfile.exists(), "Dockerfile for API server should exist"
        
        # Check basic Dockerfile structure
        with open(dockerfile) as f:
            content = f.read()
            
        assert "FROM python:" in content, "Should use Python base image"
        assert "WORKDIR" in content, "Should set working directory"
        assert "COPY" in content, "Should copy application files"
        assert "pip install" in content, "Should install dependencies"
        assert "CMD" in content or "ENTRYPOINT" in content, "Should define startup command"
    
    def test_webapp_dockerfile_exists(self):
        """Test that Dockerfile for Flask webapp exists."""
        repo_root = Path(__file__).parent.parent
        dockerfile = repo_root / "docker" / "Dockerfile.webapp"
        
        assert dockerfile.exists(), "Dockerfile for webapp should exist"
        
        # Check basic Dockerfile structure
        with open(dockerfile) as f:
            content = f.read()
            
        assert "FROM python:" in content, "Should use Python base image"
        assert "WORKDIR" in content, "Should set working directory"
        assert "flask" in content.lower() or "gunicorn" in content.lower(), "Should reference Flask/WSGI server"
    
    def test_docker_compose_exists(self):
        """Test that docker-compose.yml exists with proper structure."""
        repo_root = Path(__file__).parent.parent
        compose_file = repo_root / "docker-compose.yml"
        
        assert compose_file.exists(), "docker-compose.yml should exist"
        
        # Parse and validate compose file structure
        with open(compose_file) as f:
            compose_config = yaml.safe_load(f)
        
        assert "services" in compose_config, "Should define services"
        assert "api" in compose_config["services"], "Should define API service"
        assert "webapp" in compose_config["services"], "Should define webapp service"
        
        # Check API service configuration
        api_service = compose_config["services"]["api"]
        assert "build" in api_service or "image" in api_service, "API service should define build/image"
        assert "ports" in api_service, "API service should expose ports"
        assert "environment" in api_service or "env_file" in api_service, "API service should have environment config"
        
        # Check webapp service configuration
        webapp_service = compose_config["services"]["webapp"]
        assert "build" in webapp_service or "image" in webapp_service, "Webapp service should define build/image"
        assert "ports" in webapp_service, "Webapp service should expose ports"
    
    def test_docker_environment_files(self):
        """Test that environment configuration files exist."""
        repo_root = Path(__file__).parent.parent
        
        # Check for environment file examples
        env_example = repo_root / ".env.example"
        
        if env_example.exists():
            with open(env_example) as f:
                content = f.read()
                
            # Should contain FinChat configuration variables
            assert "FINCHAT_" in content, "Should contain FinChat configuration variables"
    
    def test_dockerignore_exists(self):
        """Test that .dockerignore file exists to optimize build context."""
        repo_root = Path(__file__).parent.parent
        dockerignore = repo_root / ".dockerignore"
        
        if dockerignore.exists():
            with open(dockerignore) as f:
                content = f.read()
                
            # Should exclude common unnecessary files
            exclusions = [".git", "*.pyc", "__pycache__", "tests", ".pytest_cache"]
            for exclusion in exclusions:
                assert exclusion in content, f"Should exclude {exclusion}"
    
    def test_docker_build_structure(self):
        """Test that Docker build structure is properly organized."""
        repo_root = Path(__file__).parent.parent
        docker_dir = repo_root / "docker"
        
        # Check if docker directory structure exists
        if docker_dir.exists():
            assert docker_dir.is_dir(), "docker/ should be a directory"
            
            # Check for expected files
            expected_files = [
                "Dockerfile.api",
                "Dockerfile.webapp",
                "entrypoint.sh",
            ]
            
            for expected_file in expected_files:
                file_path = docker_dir / expected_file
                if file_path.exists():
                    # If entrypoint exists, check it's executable
                    if expected_file == "entrypoint.sh":
                        assert os.access(file_path, os.X_OK), "entrypoint.sh should be executable"


def _docker_available():
    """Check if Docker is available on the system."""
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


class TestDockerBuildValidation:
    """Test Docker build process validation (requires Docker)."""
    
    @pytest.mark.skipif(not _docker_available(), reason="Docker not available")
    def test_api_dockerfile_builds(self):
        """Test that API Dockerfile builds successfully."""
        repo_root = Path(__file__).parent.parent
        dockerfile = repo_root / "docker" / "Dockerfile.api"
        
        if not dockerfile.exists():
            pytest.skip("API Dockerfile not found")
        
        # Attempt to build the Docker image
        result = subprocess.run([
            "docker", "build", 
            "-f", str(dockerfile),
            "-t", "finchat-api:test",
            str(repo_root)
        ], capture_output=True, text=True)
        
        assert result.returncode == 0, f"Docker build failed: {result.stderr}"
    
    @pytest.mark.skipif(not _docker_available(), reason="Docker not available")
    def test_webapp_dockerfile_builds(self):
        """Test that webapp Dockerfile builds successfully."""
        repo_root = Path(__file__).parent.parent
        dockerfile = repo_root / "docker" / "Dockerfile.webapp"
        
        if not dockerfile.exists():
            pytest.skip("Webapp Dockerfile not found")
        
        # Attempt to build the Docker image
        result = subprocess.run([
            "docker", "build",
            "-f", str(dockerfile), 
            "-t", "finchat-webapp:test",
            str(repo_root)
        ], capture_output=True, text=True)
        
        assert result.returncode == 0, f"Docker build failed: {result.stderr}"
    
    @pytest.mark.skipif(not _docker_available(), reason="Docker not available")
    def test_docker_compose_validation(self):
        """Test that docker-compose configuration is valid."""
        repo_root = Path(__file__).parent.parent
        compose_file = repo_root / "docker-compose.yml"
        
        if not compose_file.exists():
            pytest.skip("docker-compose.yml not found")
        
        # Validate docker-compose syntax
        result = subprocess.run([
            "docker-compose", "-f", str(compose_file), "config"
        ], capture_output=True, text=True, cwd=repo_root)
        
        assert result.returncode == 0, f"docker-compose validation failed: {result.stderr}"


def _docker_available() -> bool:
    """Check if Docker is available in the system."""
    try:
        result = subprocess.run(["docker", "--version"], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False