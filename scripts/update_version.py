#!/usr/bin/env python3
"""
Script to update version numbers across the project for semantic-release.
"""

import argparse
import re
import sys
from pathlib import Path


def update_pyproject_toml(version: str, project_root: Path) -> bool:
    """Update version in pyproject.toml."""
    pyproject_path = project_root / "pyproject.toml"
    
    if not pyproject_path.exists():
        print(f"Error: {pyproject_path} not found")
        return False
    
    content = pyproject_path.read_text()
    
    # Update version in [project] section
    pattern = r'(version\s*=\s*["\'])([^"\']+)(["\'])'
    replacement = rf'\g<1>{version}\g<3>'
    
    new_content = re.sub(pattern, replacement, content)
    
    if new_content == content:
        print("Warning: No version pattern found in pyproject.toml")
        return False
    
    pyproject_path.write_text(new_content)
    print(f"Updated pyproject.toml version to {version}")
    return True


def update_package_init(version: str, project_root: Path) -> bool:
    """Update version in package __init__.py."""
    init_path = project_root / "src" / "finchat_sec_qa" / "__init__.py"
    
    if not init_path.exists():
        print(f"Error: {init_path} not found")
        return False
    
    content = init_path.read_text()
    
    # Look for __version__ = "..." pattern
    pattern = r'(__version__\s*=\s*["\'])([^"\']+)(["\'])'
    replacement = rf'\g<1>{version}\g<3>'
    
    new_content = re.sub(pattern, replacement, content)
    
    if new_content == content:
        # If pattern not found, add it
        lines = content.splitlines()
        if not any("__version__" in line for line in lines):
            # Add version after module docstring if it exists
            docstring_end = 0
            in_docstring = False
            
            for i, line in enumerate(lines):
                if line.strip().startswith('"""') or line.strip().startswith("'''"):
                    if not in_docstring:
                        in_docstring = True
                    elif line.strip().endswith('"""') or line.strip().endswith("'''"):
                        docstring_end = i
                        break
            
            lines.insert(docstring_end + 1, f'__version__ = "{version}"')
            new_content = "\n".join(lines) + "\n"
        else:
            print("Warning: No version pattern found in __init__.py")
            return False
    
    init_path.write_text(new_content)
    print(f"Updated __init__.py version to {version}")
    return True


def update_docker_labels(version: str, project_root: Path) -> bool:
    """Update version labels in Dockerfiles."""
    docker_dir = project_root / "docker"
    updated_files = []
    
    if not docker_dir.exists():
        print("Warning: docker directory not found")
        return True
    
    for dockerfile in docker_dir.glob("Dockerfile.*"):
        content = dockerfile.read_text()
        
        # Update LABEL version
        pattern = r'(LABEL\s+version\s*=\s*["\'])([^"\']+)(["\'])'
        replacement = rf'\g<1>{version}\g<3>'
        
        new_content = re.sub(pattern, replacement, content)
        
        if new_content != content:
            dockerfile.write_text(new_content)
            updated_files.append(dockerfile.name)
    
    if updated_files:
        print(f"Updated Docker files: {', '.join(updated_files)}")
    
    return True


def validate_version(version: str) -> bool:
    """Validate that version follows semantic versioning."""
    pattern = r'^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9\-\.]+))?(?:\+([a-zA-Z0-9\-\.]+))?$'
    return bool(re.match(pattern, version))


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Update project version numbers")
    parser.add_argument("version", help="New version number (semantic versioning)")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory (default: current directory)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without making changes"
    )
    
    args = parser.parse_args()
    
    if not validate_version(args.version):
        print(f"Error: Invalid version format: {args.version}")
        print("Version must follow semantic versioning (e.g., 1.2.3, 1.2.3-beta.1)")
        sys.exit(1)
    
    if args.dry_run:
        print(f"DRY RUN: Would update version to {args.version}")
        return
    
    print(f"Updating project version to {args.version}")
    
    success = True
    
    # Update pyproject.toml
    if not update_pyproject_toml(args.version, args.project_root):
        success = False
    
    # Update package __init__.py
    if not update_package_init(args.version, args.project_root):
        success = False
    
    # Update Docker files
    if not update_docker_labels(args.version, args.project_root):
        success = False
    
    if success:
        print(f"Successfully updated all version references to {args.version}")
    else:
        print("Some version updates failed")
        sys.exit(1)


if __name__ == "__main__":
    main()