from finchat_sec_qa import cli
from finchat_sec_qa.edgar_client import FilingMetadata
from finchat_sec_qa.risk_intelligence import RiskAssessment


def test_ingest_subcommand(monkeypatch, tmp_path):
    calls = {}

    class DummyClient:
        def __init__(self, ua):
            calls['ua'] = ua
        def get_recent_filings(self, ticker, form_type='10-K', limit=1):
            calls['ticker'] = ticker
            return [FilingMetadata(cik='1', accession_no='1', form_type=form_type, filing_date=None, document_url='http://x')]
        def download_filing(self, filing, dest):
            (tmp_path / 'f.html').write_text('a')
            calls['dest'] = dest
            return tmp_path / 'f.html'

    monkeypatch.setattr(cli, 'EdgarClient', DummyClient)
    cli.main(['ingest', 'AAPL', '--dest', str(tmp_path)])
    assert calls['ticker'] == 'AAPL'
    assert (tmp_path / 'f.html').exists()


def test_risk_subcommand(monkeypatch, tmp_path, capsys):
    text = tmp_path / 't.txt'
    text.write_text('lawsuit')

    class DummyRisk:
        def assess(self, txt):
            return RiskAssessment(text=txt, sentiment=-0.5, flags=['litigation'])

    monkeypatch.setattr(cli, 'RiskAnalyzer', lambda: DummyRisk())
    cli.main(['risk', str(text)])
    out = capsys.readouterr().out
    assert 'litigation' in out
