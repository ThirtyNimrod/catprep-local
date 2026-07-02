import json
import sys

from core.token_summary import format_summary_table, parse_token_log
from scripts.summarize_tokens import main as summarize_tokens_main


def test_parse_token_log_and_format(tmp_path):
    log_file = tmp_path / "token_usage.log"
    log_file.write_text(
        "[router_agent] Prompt: 10, Completion: 20, Total: 30\n"
        "[study_plan] Prompt: 50, Completion: 100, Total: 150\n"
        "[router_agent] No usage found. Debug info: None\n"
    )

    stats = parse_token_log(log_file)
    assert "router_agent" in stats
    assert stats["router_agent"]["calls"] == 1
    assert stats["router_agent"]["total_tokens"] == 30
    assert stats["study_plan"]["calls"] == 1
    assert stats["study_plan"]["total_tokens"] == 150
    assert stats["_untracked"]["calls"] == 1

    table = format_summary_table(stats)
    assert "study_plan" in table
    assert "router_agent" in table
    assert "_untracked" in table


def test_parse_token_log_missing_file_returns_empty_dict(tmp_path):
    assert parse_token_log(tmp_path / "does_not_exist.log") == {}


def test_format_summary_table_empty_stats():
    assert format_summary_table({}) == "No token usage recorded."


def test_summarize_tokens_cli_text_output(tmp_path, capsys, monkeypatch):
    log_file = tmp_path / "token_usage.log"
    log_file.write_text("[router_agent] Prompt: 10, Completion: 20, Total: 30\n")
    monkeypatch.setattr(sys, "argv", ["summarize_tokens.py", "--log-file", str(log_file)])

    summarize_tokens_main()

    out = capsys.readouterr().out
    assert "router_agent" in out
    assert "30" in out


def test_summarize_tokens_cli_json_output(tmp_path, capsys, monkeypatch):
    log_file = tmp_path / "token_usage.log"
    log_file.write_text("[router_agent] Prompt: 10, Completion: 20, Total: 30\n")
    monkeypatch.setattr(sys, "argv", ["summarize_tokens.py", "--log-file", str(log_file), "--json"])

    summarize_tokens_main()

    data = json.loads(capsys.readouterr().out)
    assert data["router_agent"]["total_tokens"] == 30
