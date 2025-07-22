"""Tests for contributor documentation completeness and quality."""
from __future__ import annotations

import pytest
import re
from pathlib import Path


class TestContributorDocumentation:
    """Test contributor documentation structure and content."""
    
    def test_contributing_guide_exists(self):
        """Test that comprehensive CONTRIBUTING.md exists."""
        repo_root = Path(__file__).parent.parent
        contributing_file = repo_root / "CONTRIBUTING.md"
        
        assert contributing_file.exists(), "CONTRIBUTING.md should exist"
        
        with open(contributing_file) as f:
            content = f.read()
        
        # Check for essential sections
        assert "# Contributing" in content, "Should have main title"
        assert "## Development Setup" in content, "Should have development setup section"
        assert "## Testing" in content, "Should have testing section"
        assert "## Pull Request Process" in content, "Should have PR process section"
        assert "## Code Style" in content, "Should have code style section"
    
    def test_development_setup_guide_exists(self):
        """Test that detailed development setup guide exists."""
        repo_root = Path(__file__).parent.parent
        setup_guide = repo_root / "docs" / "DEVELOPMENT_SETUP.md"
        
        assert setup_guide.exists(), "Development setup guide should exist"
        
        with open(setup_guide) as f:
            content = f.read()
        
        # Check for essential setup instructions
        assert "# Development Setup Guide" in content, "Should have main title"
        assert "## Prerequisites" in content, "Should list prerequisites"
        assert "## Local Development" in content, "Should have local dev instructions"
        assert "## Testing" in content, "Should have testing instructions"
        assert "## Docker Development" in content, "Should have Docker instructions"
    
    def test_issue_templates_exist(self):
        """Test that GitHub issue templates exist."""
        repo_root = Path(__file__).parent.parent
        templates_dir = repo_root / ".github" / "ISSUE_TEMPLATE"
        
        assert templates_dir.exists(), "Issue templates directory should exist"
        
        # Check for specific template files
        bug_template = templates_dir / "bug_report.yml"
        feature_template = templates_dir / "feature_request.yml"
        
        assert bug_template.exists(), "Bug report template should exist"
        assert feature_template.exists(), "Feature request template should exist"
        
        # Check bug report template structure
        with open(bug_template) as f:
            bug_content = f.read()
        
        assert "name: Bug Report" in bug_content, "Bug template should have proper name"
        assert "description:" in bug_content, "Should have description field"
        assert "body:" in bug_content, "Should have body section"
        
        # Check feature request template structure  
        with open(feature_template) as f:
            feature_content = f.read()
        
        assert "name: Feature Request" in feature_content, "Feature template should have proper name"
        assert "description:" in feature_content, "Should have description field"
    
    def test_pull_request_template_exists(self):
        """Test that pull request template exists."""
        repo_root = Path(__file__).parent.parent
        pr_template = repo_root / ".github" / "pull_request_template.md"
        
        assert pr_template.exists(), "PR template should exist"
        
        with open(pr_template) as f:
            content = f.read()
        
        # Check for essential PR template sections
        assert "## Summary" in content, "Should have summary section"
        assert "## Changes Made" in content, "Should have changes section"
        assert "## Testing" in content, "Should have testing section"
        assert "## Checklist" in content, "Should have checklist section"
    
    def test_contributing_guide_content_quality(self):
        """Test that CONTRIBUTING.md has comprehensive content."""
        repo_root = Path(__file__).parent.parent
        contributing_file = repo_root / "CONTRIBUTING.md"
        
        with open(contributing_file) as f:
            content = f.read()
        
        # Check for specific helpful content
        assert "python" in content.lower(), "Should mention Python setup"
        assert "virtual environment" in content.lower() or "venv" in content.lower(), "Should mention virtual environments"
        assert "pytest" in content.lower(), "Should mention pytest"
        assert "docker" in content.lower(), "Should mention Docker"
        assert "sdk" in content.lower(), "Should mention SDK"
    
    def test_setup_guide_completeness(self):
        """Test that development setup guide is comprehensive."""
        repo_root = Path(__file__).parent.parent
        setup_guide = repo_root / "docs" / "DEVELOPMENT_SETUP.md"
        
        if setup_guide.exists():
            with open(setup_guide) as f:
                content = f.read()
            
            # Check for comprehensive setup instructions
            assert "git clone" in content.lower(), "Should have git clone instructions"
            assert "pip install" in content.lower(), "Should have pip install instructions"
            assert "docker-compose" in content.lower(), "Should mention docker-compose"
            assert "environment" in content.lower(), "Should mention environment setup"
    
    def test_code_of_conduct_exists(self):
        """Test that code of conduct exists."""
        repo_root = Path(__file__).parent.parent
        code_of_conduct = repo_root / "CODE_OF_CONDUCT.md"
        
        # This is optional but recommended
        if code_of_conduct.exists():
            with open(code_of_conduct) as f:
                content = f.read()
            
            assert "Code of Conduct" in content, "Should have proper title"
            assert "behavior" in content.lower(), "Should address behavior"
    
    def test_security_policy_exists(self):
        """Test that security policy exists."""
        repo_root = Path(__file__).parent.parent
        security_policy = repo_root / "SECURITY.md"
        
        # This is optional but recommended for security-focused projects
        if security_policy.exists():
            with open(security_policy) as f:
                content = f.read()
            
            assert "Security Policy" in content, "Should have proper title"
            assert "vulnerability" in content.lower(), "Should mention vulnerabilities"


class TestDocumentationQuality:
    """Test overall documentation quality and consistency."""
    
    def test_readme_references_contributing(self):
        """Test that README references contributing guide."""
        repo_root = Path(__file__).parent.parent
        readme_file = repo_root / "README.md"
        
        if readme_file.exists():
            with open(readme_file) as f:
                content = f.read()
            
            # Should reference contributing
            assert "contribut" in content.lower(), "README should mention contributing"
    
    def test_documentation_cross_references(self):
        """Test that documentation files properly cross-reference each other."""
        repo_root = Path(__file__).parent.parent
        contributing_file = repo_root / "CONTRIBUTING.md"
        
        if contributing_file.exists():
            with open(contributing_file) as f:
                content = f.read()
            
            # Should reference other documentation
            assert "DEVELOPMENT_SETUP" in content or "development setup" in content.lower(), "Should reference setup guide"
    
    def test_all_markdown_files_have_proper_headers(self):
        """Test that all markdown files have proper header structure."""
        repo_root = Path(__file__).parent.parent
        
        # Find all markdown files
        md_files = list(repo_root.glob("**/*.md"))
        
        for md_file in md_files:
            # Skip certain directories
            if any(part in str(md_file) for part in [".git", "node_modules", "venv"]):
                continue
                
            with open(md_file) as f:
                content = f.read()
            
            # Should start with h1 header
            lines = content.strip().split('\n')
            if lines:
                first_content_line = next((line for line in lines if line.strip()), "")
                # Allow for front matter or other content, but markdown files should generally have headers
                has_h1 = any(line.startswith('# ') for line in lines[:10])  # Check first 10 lines
                
                # This is a soft requirement - not all files need h1 headers
                if md_file.name not in ["pull_request_template.md"]:
                    if content.strip() and len(content.strip()) > 50:  # Only check substantial files
                        assert has_h1 or "template" in md_file.name.lower(), f"{md_file.name} should have h1 header"