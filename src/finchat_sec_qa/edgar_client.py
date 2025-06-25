from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List, Optional

import requests


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
        self, user_agent: str, session: Optional[requests.Session] = None
    ) -> None:
        if not user_agent:
            raise ValueError("user_agent must be provided for SEC requests")
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    def _get(self, url: str) -> requests.Response:
        response = self.session.get(url)
        response.raise_for_status()
        return response

    def ticker_to_cik(self, ticker: str) -> str:
        """Return the CIK (Central Index Key) for a given ticker symbol."""
        ticker = ticker.upper()
        mapping_url = f"{self.BASE_URL}/files/company_tickers.json"
        data = self._get(mapping_url).json()
        for entry in data.values():
            if entry["ticker"].upper() == ticker:
                return str(entry["cik_str"]).zfill(10)
        raise ValueError(f"Ticker '{ticker}' not found")

    def get_recent_filings(
        self, ticker: str, form_type: str = "10-K", limit: int = 10
    ) -> List[FilingMetadata]:
        """Fetch metadata for the most recent filings of a company."""
        cik = self.ticker_to_cik(ticker)
        url = f"{self.BASE_URL}/submissions/CIK{cik}.json"
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
            doc_url = f"{self.BASE_URL}/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{link}"
            results.append(
                FilingMetadata(
                    cik=cik,
                    accession_no=accession,
                    form_type=form,
                    filing_date=date.fromisoformat(filed),
                    document_url=doc_url,
                )
            )
            if len(results) >= limit:
                break
        return results

    def download_filing(self, filing: FilingMetadata, destination: Path) -> Path:
        """Download a filing document to the given destination directory."""
        destination.mkdir(parents=True, exist_ok=True)
        filename = destination / f"{filing.accession_no}-{filing.form_type}.html"
        if not filename.exists():
            response = self._get(filing.document_url)
            filename.write_bytes(response.content)
        return filename
