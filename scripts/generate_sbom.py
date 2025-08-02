#!/usr/bin/env python3
"""Generate Software Bill of Materials (SBOM) for FinChat-SEC-QA.

This script generates an SBOM in SPDX format that includes all dependencies
and their license information for compliance and security tracking.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


def get_installed_packages() -> List[Dict[str, str]]:
    """Get list of installed packages with versions."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            check=True
        )
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error getting package list: {e}")
        return []


def get_package_info(package_name: str) -> Dict[str, Any]:
    """Get detailed package information including license."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", package_name],
            capture_output=True,
            text=True,
            check=True
        )
        
        info = {}
        for line in result.stdout.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                info[key.strip()] = value.strip()
        
        return info
    except subprocess.CalledProcessError:
        return {}


def generate_spdx_sbom() -> Dict[str, Any]:
    """Generate SBOM in SPDX format."""
    packages = get_installed_packages()
    
    sbom = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "FinChat-SEC-QA-SBOM",
        "documentNamespace": f"https://github.com/danieleschmidt/finchat-sec-qa/sbom-{datetime.now().isoformat()}",
        "creationInfo": {
            "created": datetime.now().isoformat(),
            "creators": ["Tool: generate_sbom.py"],
            "licenseListVersion": "3.21"
        },
        "packages": [],
        "relationships": []
    }
    
    # Add main package
    main_package = {
        "SPDXID": "SPDXRef-Package-FinChat-SEC-QA",
        "name": "finchat-sec-qa",
        "downloadLocation": "https://github.com/danieleschmidt/finchat-sec-qa",
        "filesAnalyzed": False,
        "licenseConcluded": "MIT",
        "licenseDeclared": "MIT",
        "copyrightText": "Copyright (c) 2024 FinChat-SEC-QA Contributors"
    }
    sbom["packages"].append(main_package)
    
    # Add dependencies
    for i, package in enumerate(packages):
        if package["name"] == "finchat-sec-qa":
            continue
            
        package_info = get_package_info(package["name"])
        license_info = package_info.get("License", "NOASSERTION")
        
        # Clean up common license variations
        if license_info.lower() in ["unknown", "", "none"]:
            license_info = "NOASSERTION"
        elif "bsd" in license_info.lower():
            license_info = "BSD-3-Clause"
        elif "mit" in license_info.lower():
            license_info = "MIT"
        elif "apache" in license_info.lower():
            license_info = "Apache-2.0"
        
        pkg = {
            "SPDXID": f"SPDXRef-Package-{package['name']}-{i}",
            "name": package["name"],
            "versionInfo": package["version"],
            "downloadLocation": package_info.get("Home-page", "NOASSERTION"),
            "filesAnalyzed": False,
            "licenseConcluded": license_info,
            "licenseDeclared": license_info,
            "copyrightText": "NOASSERTION"
        }
        
        # Add package description if available
        if "Summary" in package_info:
            pkg["description"] = package_info["Summary"]
        
        sbom["packages"].append(pkg)
        
        # Add dependency relationship
        relationship = {
            "spdxElementId": "SPDXRef-Package-FinChat-SEC-QA",
            "relationshipType": "DEPENDS_ON",
            "relatedSpdxElement": pkg["SPDXID"]
        }
        sbom["relationships"].append(relationship)
    
    return sbom


def generate_cyclonedx_sbom() -> Dict[str, Any]:
    """Generate SBOM in CycloneDX format."""
    packages = get_installed_packages()
    
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.4",
        "serialNumber": f"urn:uuid:finchat-sec-qa-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "tools": [
                {
                    "vendor": "FinChat-SEC-QA",
                    "name": "generate_sbom.py",
                    "version": "1.0.0"
                }
            ],
            "component": {
                "type": "application",
                "bom-ref": "pkg:pypi/finchat-sec-qa",
                "name": "finchat-sec-qa",
                "version": "1.4.9",
                "description": "RAG agent for SEC filing analysis with citation tracking",
                "licenses": [{"license": {"id": "MIT"}}]
            }
        },
        "components": []
    }
    
    # Add dependencies
    for package in packages:
        if package["name"] == "finchat-sec-qa":
            continue
            
        package_info = get_package_info(package["name"])
        
        component = {
            "type": "library",
            "bom-ref": f"pkg:pypi/{package['name']}@{package['version']}",
            "name": package["name"],
            "version": package["version"],
            "purl": f"pkg:pypi/{package['name']}@{package['version']}"
        }
        
        # Add description if available
        if "Summary" in package_info:
            component["description"] = package_info["Summary"]
        
        # Add license if available
        license_info = package_info.get("License", "")
        if license_info and license_info.lower() not in ["unknown", "", "none"]:
            component["licenses"] = [{"license": {"name": license_info}}]
        
        # Add homepage if available
        if "Home-page" in package_info:
            component["externalReferences"] = [
                {
                    "type": "website",
                    "url": package_info["Home-page"]
                }
            ]
        
        sbom["components"].append(component)
    
    return sbom


def main():
    """Main function to generate SBOM files."""
    output_dir = Path(".")
    
    print("Generating Software Bill of Materials (SBOM)...")
    
    # Generate SPDX format SBOM
    print("Generating SPDX format SBOM...")
    spdx_sbom = generate_spdx_sbom()
    spdx_file = output_dir / "sbom.spdx.json"
    with open(spdx_file, 'w') as f:
        json.dump(spdx_sbom, f, indent=2)
    print(f"SPDX SBOM saved to: {spdx_file}")
    
    # Generate CycloneDX format SBOM
    print("Generating CycloneDX format SBOM...")
    cyclonedx_sbom = generate_cyclonedx_sbom()
    cyclonedx_file = output_dir / "sbom.cyclonedx.json"
    with open(cyclonedx_file, 'w') as f:
        json.dump(cyclonedx_sbom, f, indent=2)
    print(f"CycloneDX SBOM saved to: {cyclonedx_file}")
    
    print(f"\nGenerated SBOMs for {len(spdx_sbom['packages'])} packages")
    print("\nSBOM files can be used for:")
    print("- Supply chain security analysis")
    print("- License compliance tracking") 
    print("- Vulnerability scanning")
    print("- Regulatory compliance")


if __name__ == "__main__":
    main()