"""Secure file operations to prevent path traversal and other file-based attacks."""

import os
import logging
from pathlib import Path
from typing import List, Union, Optional

logger = logging.getLogger(__name__)


def get_allowed_directories() -> List[Path]:
    """Get list of directories where file operations are allowed.
    
    Returns:
        List of Path objects representing allowed directories
    """
    return [
        Path.home() / ".cache" / "finchat_sec_qa",  # Main cache directory
        Path("/tmp") / "finchat_sec_qa",  # Temp directory for testing
    ]


def validate_file_path(
    file_path: Union[str, Path], 
    allowed_dirs: Optional[List[Path]] = None
) -> Path:
    """Validate that a file path is safe and within allowed directories.
    
    Args:
        file_path: The file path to validate
        allowed_dirs: Optional list of allowed directories (defaults to get_allowed_directories())
        
    Returns:
        Normalized Path object if valid
        
    Raises:
        ValueError: If path is invalid, contains traversal attempts, or is outside allowed dirs
    """
    if allowed_dirs is None:
        allowed_dirs = get_allowed_directories()
    
    # Convert to Path and resolve to normalize
    try:
        path = Path(file_path).resolve()
    except (OSError, ValueError) as e:
        raise ValueError(f"Invalid path: {e}")
    
    # Check for absolute paths outside allowed areas
    if path.is_absolute():
        # Check if path is within any allowed directory
        path_allowed = False
        for allowed_dir in allowed_dirs:
            try:
                allowed_dir_resolved = allowed_dir.resolve()
                # Check if path is within this allowed directory
                path.relative_to(allowed_dir_resolved)
                path_allowed = True
                break
            except ValueError:
                # path is not relative to this allowed_dir, continue checking
                continue
        
        if not path_allowed:
            raise ValueError(f"Path not in allowed directories: {path}")
    else:
        # Relative paths - resolve against first allowed directory
        if not allowed_dirs:
            raise ValueError("No allowed directories specified for relative path")
        
        base_dir = allowed_dirs[0].resolve()
        path = (base_dir / file_path).resolve()
        
        # Ensure resolved path is still within the base directory
        try:
            path.relative_to(base_dir)
        except ValueError:
            raise ValueError(f"Path traversal detected: {file_path} resolves outside allowed directory")
    
    # Additional security checks
    path_str = str(path)
    
    # Check for suspicious patterns
    suspicious_patterns = [
        '../', '/etc/', '/proc/', '/sys/', '/dev/', '/var/log/',
        'C:\\Windows\\', 'C:\\System32\\', '\\Windows\\', '\\System32\\',
        '~/', '$HOME', '%USERPROFILE%'
    ]
    
    for pattern in suspicious_patterns:
        if pattern in path_str:
            logger.error("Blocked suspicious path pattern: %s in %s", pattern, path_str)
            raise ValueError(f"Suspicious path pattern detected: {pattern} in {path_str}")
    
    # Check if it's a symbolic link (potential symlink attack)
    if path.exists() and path.is_symlink():
        # Resolve the symlink and check if target is in allowed directories
        target = path.readlink()
        if target.is_absolute():
            # Check if symlink target is in allowed directories  
            target_allowed = False
            for allowed_dir in allowed_dirs:
                try:
                    target.relative_to(allowed_dir.resolve())
                    target_allowed = True
                    break
                except ValueError:
                    continue
            
            if not target_allowed:
                raise ValueError(f"Symbolic link target not in allowed directories: {target}")
    
    return path


def safe_read_file(
    file_path: Union[str, Path], 
    allowed_dirs: Optional[List[Path]] = None,
    encoding: str = 'utf-8'
) -> str:
    """Safely read a file with path validation.
    
    Args:
        file_path: Path to the file to read
        allowed_dirs: Optional list of allowed directories
        encoding: File encoding (default: utf-8)
        
    Returns:
        File contents as string
        
    Raises:
        ValueError: If path validation fails
        FileNotFoundError: If file doesn't exist
        PermissionError: If no permission to read file
        UnicodeDecodeError: If file encoding is invalid
    """
    # Validate the path first
    validated_path = validate_file_path(file_path, allowed_dirs)
    
    try:
        # Read the file
        content = validated_path.read_text(encoding=encoding)
        logger.debug("Successfully read file: %s (%d chars)", validated_path, len(content))
        return content
        
    except FileNotFoundError:
        logger.error("File not found: %s", validated_path)
        raise
        
    except PermissionError:
        logger.error("Permission denied reading file: %s", validated_path)
        raise
        
    except UnicodeDecodeError as e:
        logger.error("Encoding error reading file %s: %s", validated_path, e)
        raise


def safe_write_file(
    file_path: Union[str, Path],
    content: str,
    allowed_dirs: Optional[List[Path]] = None,
    encoding: str = 'utf-8'
) -> None:
    """Safely write to a file with path validation.
    
    Args:
        file_path: Path to the file to write
        content: Content to write
        allowed_dirs: Optional list of allowed directories
        encoding: File encoding (default: utf-8)
        
    Raises:
        ValueError: If path validation fails
        PermissionError: If no permission to write file
        OSError: If write operation fails
    """
    # Validate the path first
    validated_path = validate_file_path(file_path, allowed_dirs)
    
    # Ensure parent directory exists
    validated_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Write the file
        validated_path.write_text(content, encoding=encoding)
        logger.debug("Successfully wrote file: %s (%d chars)", validated_path, len(content))
        
    except PermissionError:
        logger.error("Permission denied writing file: %s", validated_path)
        raise
        
    except OSError as e:
        logger.error("Error writing file %s: %s", validated_path, e)
        raise


def safe_file_exists(
    file_path: Union[str, Path],
    allowed_dirs: Optional[List[Path]] = None
) -> bool:
    """Safely check if a file exists with path validation.
    
    Args:
        file_path: Path to check
        allowed_dirs: Optional list of allowed directories
        
    Returns:
        True if file exists and path is valid, False otherwise
    """
    try:
        validated_path = validate_file_path(file_path, allowed_dirs)
        return validated_path.exists()
    except (ValueError, OSError):
        return False