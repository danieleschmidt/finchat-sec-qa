import logging
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


def test_cli_logging_integration(monkeypatch, tmp_path, caplog):
    """Test that CLI commands produce appropriate log messages."""
    
    # Test ingest command logging
    class DummyClient:
        def __init__(self, ua):
            pass
        def get_recent_filings(self, ticker, form_type='10-K', limit=1):
            return [FilingMetadata(cik='1', accession_no='1', form_type=form_type, filing_date=None, document_url='http://x')]
        def download_filing(self, filing, dest):
            (tmp_path / 'f.html').write_text('test content')
            return tmp_path / 'f.html'

    monkeypatch.setattr(cli, 'EdgarClient', DummyClient)
    
    with caplog.at_level(logging.INFO):
        cli.main(['ingest', 'AAPL', '--dest', str(tmp_path)])
    
    # Check that key log messages are present
    log_messages = [record.message for record in caplog.records]
    assert any('Starting ingest command for ticker: AAPL' in msg for msg in log_messages)
    assert any('Found 1 filings to download' in msg for msg in log_messages)
    assert any('Ingest command completed successfully' in msg for msg in log_messages)


def test_risk_command_logging(monkeypatch, tmp_path, caplog):
    """Test that risk command produces appropriate log messages."""
    text_file = tmp_path / 'test.txt'
    text_file.write_text('This is a test document with some content.')

    class DummyRisk:
        def assess(self, txt):
            return RiskAssessment(text=txt, sentiment=-0.2, flags=['test-flag'])

    monkeypatch.setattr(cli, 'RiskAnalyzer', lambda: DummyRisk())
    
    with caplog.at_level(logging.INFO):
        cli.main(['risk', str(text_file)])
    
    # Check that key log messages are present
    log_messages = [record.message for record in caplog.records]
    assert any(f'Starting risk analysis for file: {text_file}' in msg for msg in log_messages)
    assert any('Risk assessment completed with 1 flags' in msg for msg in log_messages)
    assert any('Risk command completed successfully' in msg for msg in log_messages)
