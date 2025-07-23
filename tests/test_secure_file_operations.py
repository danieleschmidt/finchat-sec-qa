"""Tests for secure file operations to prevent path traversal attacks."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch

from finchat_sec_qa.file_security import safe_read_file, validate_file_path


def test_safe_read_file_normal_path():
    """Test that safe_read_file works with normal file paths."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("test content")
        temp_path = f.name
    
    try:
        content = safe_read_file(temp_path)
        assert content == "test content"
    finally:
        os.unlink(temp_path)


def test_safe_read_file_path_traversal_attack():
    """Test that safe_read_file prevents path traversal attacks."""
    # These should be blocked
    malicious_paths = [
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
        "/etc/passwd",
        "C:\\Windows\\System32\\drivers\\etc\\hosts",
        "~/.ssh/id_rsa",
        "./../../sensitive_file.txt"
    ]
    
    for malicious_path in malicious_paths:
        with pytest.raises(ValueError, match="Path traversal detected|Absolute path not allowed|Invalid path"):
            safe_read_file(malicious_path)


def test_safe_read_file_allows_cache_directory():
    """Test that safe_read_file allows reading from the designated cache directory."""
    cache_dir = Path.home() / ".cache" / "finchat_sec_qa"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    test_file = cache_dir / "test_file.txt"
    test_file.write_text("cache content")
    
    try:
        content = safe_read_file(str(test_file))
        assert content == "cache content"
    finally:
        test_file.unlink(missing_ok=True)


def test_safe_read_file_blocks_outside_allowed_dirs():
    """Test that safe_read_file blocks files outside allowed directories."""
    # Create a file in /tmp (should be blocked)
    with tempfile.NamedTemporaryFile(mode='w', delete=False, dir='/tmp') as f:
        f.write("should not be readable")
        temp_path = f.name
    
    try:
        with pytest.raises(ValueError, match="Path not in allowed directories"):
            safe_read_file(temp_path)
    finally:
        os.unlink(temp_path)


def test_safe_read_file_allows_temp_directory():
    """Test that safe_read_file allows reading from temp directories during testing."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("temp content")
        temp_path = f.name
    
    try:
        # Should work when explicitly allowing temp dir
        content = safe_read_file(temp_path, allowed_dirs=[Path(temp_path).parent])
        assert content == "temp content"
    finally:
        os.unlink(temp_path)


def test_validate_file_path_normal_paths():
    """Test that validate_file_path accepts normal paths."""
    cache_dir = Path.home() / ".cache" / "finchat_sec_qa"
    
    valid_paths = [
        cache_dir / "filing.html",
        cache_dir / "subdir" / "filing.html",
        str(cache_dir / "filing.txt")
    ]
    
    for path in valid_paths:
        # Should not raise exception
        validate_file_path(path)


def test_validate_file_path_rejects_traversal():
    """Test that validate_file_path rejects path traversal attempts."""
    invalid_paths = [
        "../../../etc/passwd",
        "..\\..\\windows\\system.ini",
        "/etc/passwd",
        "C:\\Windows\\System32\\config\\sam",
        "~/.ssh/id_rsa",
        "./../../config.yaml"
    ]
    
    for path in invalid_paths:
        with pytest.raises(ValueError):
            validate_file_path(path)


def test_validate_file_path_normalizes_paths():
    """Test that validate_file_path properly normalizes paths."""
    cache_dir = Path.home() / ".cache" / "finchat_sec_qa"
    
    # These should all normalize to valid paths
    paths_with_dots = [
        str(cache_dir / "." / "filing.html"),
        str(cache_dir / "subdir" / ".." / "filing.html"),
    ]
    
    for path in paths_with_dots:
        # Should not raise if normalized path is still within allowed dir
        validate_file_path(path)


def test_safe_read_file_file_not_found():
    """Test that safe_read_file handles file not found gracefully."""
    cache_dir = Path.home() / ".cache" / "finchat_sec_qa"
    nonexistent_file = cache_dir / "nonexistent.txt"
    
    with pytest.raises(FileNotFoundError):
        safe_read_file(str(nonexistent_file))


def test_safe_read_file_permission_denied():
    """Test that safe_read_file handles permission errors gracefully."""
    # This test might not work on all systems, so we'll mock it
    with patch('pathlib.Path.read_text', side_effect=PermissionError("Permission denied")):
        cache_dir = Path.home() / ".cache" / "finchat_sec_qa"
        test_file = cache_dir / "test.txt"
        
        with pytest.raises(PermissionError):
            safe_read_file(str(test_file))


def test_safe_read_file_symlink_attack():
    """Test that safe_read_file handles symbolic link attacks."""
    cache_dir = Path.home() / ".cache" / "finchat_sec_qa"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a file outside allowed area
    with tempfile.NamedTemporaryFile(mode='w', delete=False, dir='/tmp') as f:
        f.write("sensitive data")
        target_file = f.name
    
    # Create symlink pointing to it
    symlink_path = cache_dir / "malicious_link.txt"
    
    try:
        os.symlink(target_file, symlink_path)
        
        # Should detect and block symlink attacks
        with pytest.raises(ValueError, match="Symbolic link not allowed|Path not in allowed directories"):
            safe_read_file(str(symlink_path))
            
    except OSError:
        # Skip if symlinks not supported on this system
        pytest.skip("Symbolic links not supported on this system")
    finally:
        symlink_path.unlink(missing_ok=True)
        os.unlink(target_file)