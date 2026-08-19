import pytest
from click.testing import CliRunner
from knowledge.cli import cli

def test_cli_status():
    runner = CliRunner()
    result = runner.invoke(cli, ["status"])
    assert result.exit_code == 0
    assert "Knowledge Retrieval A-Z Workspace Status" in result.output

def test_cli_search():
    runner = CliRunner()
    result = runner.invoke(cli, ["search", "-q", "CAG", "--top-k", "2"])
    assert result.exit_code == 0
    assert "Executing HYBRID Search" in result.output

def test_cli_cag():
    runner = CliRunner()
    result = runner.invoke(cli, ["cag", "--preload-tokens", "16000"])
    assert result.exit_code == 0
    assert "Preloading 16000 tokens" in result.output
