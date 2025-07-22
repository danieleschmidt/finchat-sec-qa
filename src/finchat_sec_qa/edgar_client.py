from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin, quote

import asyncio
import logging
import requests
import httpx
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from .edgar_validation import validate_ticker, validate_cik, validate_accession_number


@dataclass
class FilingMetadata:
    """Basic information about a SEC filing."""

    cik: str
    accession_no: str
    form_type: str
    filing_date: date
    document_url: str


class EdgarClient:
    """Simple client for fetching filings from the SEC EDGAR system."""

    BASE_URL = "https://data.sec.gov"

    def __init__(
        self,
        user_agent: str,
        session: Optional[requests.Session] = None,
        *,
        timeout: float = 10.0,
        retries: int = 3,
        cache_dir: Path | None = None,
    ) -> None:
        if not user_agent:
            raise ValueError("user_agent must be provided for SEC requests")
        self.timeout = timeout
        self.cache_dir = Path(cache_dir or Path.home() / ".cache" / "finchat_sec_qa")
        self.session = session or requests.Session()
        retry_strategy = Retry(
            total=retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.headers.update({"User-Agent": user_agent})
        self.logger = logging.getLogger(__name__)

    def _get(self, url: str) -> requests.Response:
        self.logger.debug("GET %s", url)
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response

    def _validate_ticker(self, ticker: str) -> str:
        """Validate and sanitize ticker symbol."""
        return validate_ticker(ticker)
    
    def _validate_cik(self, cik: str) -> str:
        """Validate and sanitize CIK."""
        return validate_cik(cik)
    
    def _validate_accession_number(self, accession: str) -> str:
        """Validate and sanitize accession number."""
        return validate_accession_number(accession)

    def ticker_to_cik(self, ticker: str) -> str:
        """Return the CIK (Central Index Key) for a given ticker symbol."""
        ticker = self._validate_ticker(ticker)
        mapping_url = urljoin(self.BASE_URL, "/files/company_tickers.json")
        self.logger.debug("Resolving ticker %s", ticker)
        data = self._get(mapping_url).json()
        for entry in data.values():
            if entry["ticker"].upper() == ticker:
                cik = str(entry["cik_str"])
                return self._validate_cik(cik)
        raise ValueError(f"Ticker '{ticker}' not found")

    def get_recent_filings(
        self, ticker: str, form_type: str = "10-K", limit: int = 10
    ) -> List[FilingMetadata]:
        """Fetch metadata for the most recent filings of a company."""
        # Validate inputs
        ticker = self._validate_ticker(ticker)
        if form_type and not re.match(r'^[A-Z0-9-]{1,10}$', form_type):
            raise ValueError(f"Invalid form_type: {form_type}")
        if not isinstance(limit, int) or limit < 1 or limit > 100:
            raise ValueError("Limit must be an integer between 1 and 100")
        
        cik = self.ticker_to_cik(ticker)
        # Use urljoin for safe URL construction
        url = urljoin(self.BASE_URL, f"/submissions/CIK{cik}.json")
        self.logger.info("Fetching recent filings for %s", ticker)
        data = self._get(url).json()
        forms = data.get("filings", {}).get("recent", {})
        results: List[FilingMetadata] = []
        for accession, form, filed, link in zip(
            forms.get("accessionNumber", []),
            forms.get("form", []),
            forms.get("filingDate", []),
            forms.get("primaryDocument", []),
        ):
            if form_type and form != form_type:
                continue
            
            # Validate and sanitize all URL components
            try:
                validated_accession = self._validate_accession_number(accession)
                validated_cik = self._validate_cik(cik)
                
                # Sanitize filename - only allow alphanumeric, hyphens, dots, underscores
                if not link or not re.match(r'^[a-zA-Z0-9._-]+\.(htm|html|txt)$', link):
                    self.logger.warning("Skipping filing with invalid filename: %s", link)
                    continue
                
                # Construct URL safely using urljoin and validated components
                base_path = f"/Archives/edgar/data/{int(validated_cik)}/{validated_accession.replace('-', '')}/"
                doc_url = urljoin(self.BASE_URL, base_path + quote(link, safe='.-_'))
                
                results.append(
                    FilingMetadata(
                        cik=validated_cik,
                        accession_no=validated_accession,
                        form_type=form,
                        filing_date=date.fromisoformat(filed),
                        document_url=doc_url,
                    )
                )
            except ValueError as e:
                self.logger.warning("Skipping invalid filing data: %s", e)
                continue
                
            if len(results) >= limit:
                break
        return results

    def download_filing(
        self, filing: FilingMetadata, destination: Path | None = None
    ) -> Path:
        """Download a filing document to the given destination directory."""
        destination = destination or self.cache_dir
        destination.mkdir(parents=True, exist_ok=True)
        filename = destination / f"{filing.accession_no}-{filing.form_type}.html"
        if not filename.exists():
            self.logger.info("Downloading %s", filing.document_url)
            response = self._get(filing.document_url)
            filename.write_bytes(response.content)
        return filename


class AsyncEdgarClient:
    """Async version of EdgarClient for improved performance with concurrent requests."""

    BASE_URL = "https://data.sec.gov"

    def __init__(
        self,
        user_agent: str,
        session: Optional[httpx.AsyncClient] = None,
        *,
        timeout: float = 10.0,
        retries: int = 3,
        cache_dir: Path | None = None,
    ) -> None:
        if not user_agent:
            raise ValueError("user_agent must be provided for SEC requests")
        self.timeout = timeout
        self.cache_dir = Path(cache_dir or Path.home() / ".cache" / "finchat_sec_qa")
        
        # Create async session with retry configuration
        if session is None:
            transport = httpx.AsyncHTTPTransport(
                retries=retries,
                verify=True
            )
            self.session = httpx.AsyncClient(
                headers={"User-Agent": user_agent},
                timeout=httpx.Timeout(timeout),
                transport=transport,
                follow_redirects=True
            )
        else:
            self.session = session
            
        self.logger = logging.getLogger(__name__)

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - closes session."""
        await self.session.aclose()

    async def _get(self, url: str) -> httpx.Response:
        """Make async HTTP GET request."""
        self.logger.debug("GET %s", url)
        response = await self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response

    def _validate_ticker(self, ticker: str) -> str:
        """Validate and sanitize ticker symbol."""
        return validate_ticker(ticker)
    
    def _validate_cik(self, cik: str) -> str:
        """Validate and sanitize CIK."""
        return validate_cik(cik)
    
    def _validate_accession_number(self, accession: str) -> str:
        """Validate and sanitize accession number."""
        return validate_accession_number(accession)

    async def ticker_to_cik(self, ticker: str) -> str:
        """Return the CIK (Central Index Key) for a given ticker symbol."""
        ticker = self._validate_ticker(ticker)
        mapping_url = urljoin(self.BASE_URL, "/files/company_tickers.json")
        self.logger.debug("Resolving ticker %s", ticker)
        response = await self._get(mapping_url)
        data = response.json()
        for entry in data.values():
            if entry["ticker"].upper() == ticker:
                cik = str(entry["cik_str"])
                return self._validate_cik(cik)
        raise ValueError(f"Ticker '{ticker}' not found")

    async def get_recent_filings(
        self, ticker: str, form_type: str = "10-K", limit: int = 10
    ) -> List[FilingMetadata]:
        """Fetch metadata for the most recent filings of a company."""
        # Validate inputs
        ticker = self._validate_ticker(ticker)
        if form_type and not re.match(r'^[A-Z0-9-]{1,10}$', form_type):
            raise ValueError(f"Invalid form_type: {form_type}")
        if not isinstance(limit, int) or limit < 1 or limit > 100:
            raise ValueError("Limit must be an integer between 1 and 100")
        
        cik = await self.ticker_to_cik(ticker)
        # Use urljoin for safe URL construction
        url = urljoin(self.BASE_URL, f"/submissions/CIK{cik}.json")
        self.logger.info("Fetching recent filings for %s", ticker)
        response = await self._get(url)
        data = response.json()
        forms = data.get("filings", {}).get("recent", {})
        results: List[FilingMetadata] = []
        for accession, form, filed, link in zip(
            forms.get("accessionNumber", []),
            forms.get("form", []),
            forms.get("filingDate", []),
            forms.get("primaryDocument", []),
        ):
            if form_type and form != form_type:
                continue
            
            # Validate and sanitize all URL components
            try:
                validated_accession = self._validate_accession_number(accession)
                validated_cik = self._validate_cik(cik)
                
                # Sanitize filename - only allow alphanumeric, hyphens, dots, underscores
                if not link or not re.match(r'^[a-zA-Z0-9._-]+\.(htm|html|txt)$', link):
                    self.logger.warning("Skipping filing with invalid filename: %s", link)
                    continue
                
                # Construct URL safely using urljoin and validated components
                base_path = f"/Archives/edgar/data/{int(validated_cik)}/{validated_accession.replace('-', '')}/"
                doc_url = urljoin(self.BASE_URL, base_path + quote(link, safe='.-_'))
                
                results.append(
                    FilingMetadata(
                        cik=validated_cik,
                        accession_no=validated_accession,
                        form_type=form,
                        filing_date=date.fromisoformat(filed),
                        document_url=doc_url,
                    )
                )
            except ValueError as e:
                self.logger.warning("Skipping invalid filing data: %s", e)
                continue
                
            if len(results) >= limit:
                break
        return results

    async def download_filing(
        self, filing: FilingMetadata, destination: Path | None = None
    ) -> Path:
        """Download a filing document to the given destination directory."""
        destination = destination or self.cache_dir
        destination.mkdir(parents=True, exist_ok=True)
        filename = destination / f"{filing.accession_no}-{filing.form_type}.html"
        if not filename.exists():
            self.logger.info("Downloading %s", filing.document_url)
            response = await self._get(filing.document_url)
            filename.write_bytes(response.content)
        return filename
