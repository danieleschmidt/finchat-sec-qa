from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import logging
from pathlib import Path
import re
from typing import Dict, List, Optional
from urllib.parse import quote, urljoin

import httpx
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from .edgar_validation import validate_accession_number, validate_cik, validate_ticker


@dataclass
class FilingMetadata:
    """Basic information about a SEC filing."""

    cik: str
    accession_no: str
    form_type: str
    filing_date: date
    document_url: str


class BaseEdgarClient:
    """Base class containing common functionality for Edgar clients."""

    BASE_URL = "https://data.sec.gov"

    def _validate_user_agent(self, user_agent: str) -> None:
        """Validate user agent string."""
        if not user_agent:
            raise ValueError("user_agent must be provided for SEC requests")

    def _setup_cache_dir(self, cache_dir: Path | None) -> Path:
        """Setup cache directory with default fallback."""
        return Path(cache_dir or Path.home() / ".cache" / "finchat_sec_qa")

    def _validate_ticker(self, ticker: str) -> str:
        """Validate and sanitize ticker symbol."""
        return validate_ticker(ticker)

    def _validate_cik(self, cik: str) -> str:
        """Validate and sanitize CIK identifier."""
        return validate_cik(cik)

    def _validate_accession_number(self, accession: str) -> str:
        """Validate and sanitize accession number."""
        return validate_accession_number(accession)


class EdgarClient(BaseEdgarClient):
    """Simple client for fetching filings from the SEC EDGAR system."""

    def __init__(
        self,
        user_agent: str,
        session: Optional[requests.Session] = None,
        *,
        timeout: float = 10.0,
        retries: int = 3,
        cache_dir: Path | None = None,
    ) -> None:
        self._validate_user_agent(user_agent)
        self.timeout = timeout
        self.cache_dir = self._setup_cache_dir(cache_dir)
        self.session = session or requests.Session()
        self._ticker_cache: Optional[Dict[str, str]] = None
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

    def _load_ticker_cache(self) -> Dict[str, str]:
        """Load and cache ticker-to-CIK mapping for efficient lookups."""
        if self._ticker_cache is not None:
            return self._ticker_cache

        mapping_url = urljoin(self.BASE_URL, "/files/company_tickers.json")
        self.logger.debug("Loading ticker mapping cache")
        data = self._get(mapping_url).json()

        # Build efficient hash map for O(1) lookups
        self._ticker_cache = {
            entry["ticker"].upper(): str(entry["cik_str"])
            for entry in data.values()
        }

        self.logger.info("Loaded %d ticker mappings into cache", len(self._ticker_cache))
        return self._ticker_cache

    def ticker_to_cik(self, ticker: str) -> str:
        """Return the CIK (Central Index Key) for a given ticker symbol."""
        ticker = self._validate_ticker(ticker)
        self.logger.debug("Resolving ticker %s", ticker)

        # Use cached ticker mapping for O(1) lookup
        ticker_cache = self._load_ticker_cache()
        cik = ticker_cache.get(ticker)

        if cik is None:
            raise ValueError(f"Ticker '{ticker}' not found")

        return self._validate_cik(cik)

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


class AsyncEdgarClient(BaseEdgarClient):
    """Async version of EdgarClient for improved performance with concurrent requests."""

    def __init__(
        self,
        user_agent: str,
        session: Optional[httpx.AsyncClient] = None,
        *,
        timeout: float = 10.0,
        retries: int = 3,
        cache_dir: Path | None = None,
    ) -> None:
        self._validate_user_agent(user_agent)
        self.timeout = timeout
        self.cache_dir = self._setup_cache_dir(cache_dir)

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
        self._ticker_cache: Optional[Dict[str, str]] = None

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

    async def _load_ticker_cache(self) -> Dict[str, str]:
        """Load and cache ticker-to-CIK mapping for efficient lookups."""
        if self._ticker_cache is not None:
            return self._ticker_cache

        mapping_url = urljoin(self.BASE_URL, "/files/company_tickers.json")
        self.logger.debug("Loading ticker mapping cache")
        response = await self._get(mapping_url)
        data = response.json()

        # Build efficient hash map for O(1) lookups
        self._ticker_cache = {
            entry["ticker"].upper(): str(entry["cik_str"])
            for entry in data.values()
        }

        self.logger.info("Loaded %d ticker mappings into cache", len(self._ticker_cache))
        return self._ticker_cache

    async def ticker_to_cik(self, ticker: str) -> str:
        """Return the CIK (Central Index Key) for a given ticker symbol."""
        ticker = self._validate_ticker(ticker)
        self.logger.debug("Resolving ticker %s", ticker)

        # Use cached ticker mapping for O(1) lookup
        ticker_cache = await self._load_ticker_cache()
        cik = ticker_cache.get(ticker)

        if cik is None:
            raise ValueError(f"Ticker '{ticker}' not found")

        return self._validate_cik(cik)

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
