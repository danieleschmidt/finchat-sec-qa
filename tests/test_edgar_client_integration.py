from datetime import date
from pathlib import Path


from finchat_sec_qa.edgar_client import EdgarClient, FilingMetadata


class DummyResponse:
    def __init__(self, json_data=None, content=b"") -> None:
        self._json_data = json_data
        self.content = content

    def json(self):
        return self._json_data

    def raise_for_status(self) -> None:
        pass


def test_get_recent_filings(monkeypatch):
    mapping_json = {"0": {"ticker": "AAPL", "cik_str": 320193}}
    recent_json = {
        "filings": {
            "recent": {
                "accessionNumber": ["0000320193-24-000050"],
                "form": ["10-K"],
                "filingDate": ["2024-10-30"],
                "primaryDocument": ["aapl-20241030.htm"],
            }
        }
    }

    responses = [DummyResponse(mapping_json), DummyResponse(recent_json)]

    def fake_get(self, url):
        return responses.pop(0)

    monkeypatch.setattr(EdgarClient, "_get", fake_get)

    client = EdgarClient("ua")
    filings = client.get_recent_filings("AAPL")
    assert len(filings) == 1
    filing = filings[0]
    assert filing.cik == "0000320193"
    assert filing.form_type == "10-K"
    assert filing.accession_no == "0000320193-24-000050"
    assert filing.document_url.endswith("aapl-20241030.htm")


def test_download_filing(monkeypatch, tmp_path: Path):
    content = b"<html></html>"

    def fake_get(self, url):
        return DummyResponse(content=content)

    monkeypatch.setattr(EdgarClient, "_get", fake_get)
    client = EdgarClient("ua")
    filing = FilingMetadata(
        cik="0000320193",
        accession_no="0000320193-24-000050",
        form_type="10-K",
        filing_date=date.today(),
        document_url="https://example.com/aapl-20241030.htm",
    )
    path = client.download_filing(filing, tmp_path)
    assert path.exists()
    assert path.read_bytes() == content



def test_download_filing_cache(monkeypatch, tmp_path: Path):
    calls = []

    def fake_get(self, url):
        calls.append(url)
        return DummyResponse(content=b"c")

    monkeypatch.setattr(EdgarClient, "_get", fake_get)
    client = EdgarClient("ua", cache_dir=tmp_path)
    filing = FilingMetadata(
        cik="0000320193",
        accession_no="0000320193-24-000050",
        form_type="10-K",
        filing_date=date.today(),
        document_url="https://example.com/aapl-20241030.htm",
    )
    path1 = client.download_filing(filing)
    path2 = client.download_filing(filing)
    assert path1 == path2
    assert calls == [filing.document_url]
