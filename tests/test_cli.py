"""Tests for the agenticrag CLI entrypoint."""

from unittest.mock import patch

import pytest

from agenticrag.cli import main


def test_cli_help(capsys):
    """Test that agenticrag --help displays usage information."""
    with patch("sys.argv", ["agenticrag", "--help"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "CLI for Agentic Customer Support RAG System" in captured.out


def test_cli_serve_defaults():
    """Test that serve command invokes uvicorn with default options."""
    with (
        patch("sys.argv", ["agenticrag", "serve"]),
        patch("uvicorn.run") as mock_uvicorn_run,
    ):
        main()
        mock_uvicorn_run.assert_called_once_with(
            "agenticrag.api:app", host="0.0.0.0", port=8000, reload=False
        )


def test_cli_serve_custom_flags():
    """Test that serve command forwards custom host, port, and reload flags."""
    with (
        patch(
            "sys.argv", ["agenticrag", "serve", "--host", "127.0.0.1", "--port", "9000", "--reload"]
        ),
        patch("uvicorn.run") as mock_uvicorn_run,
    ):
        main()
        mock_uvicorn_run.assert_called_once_with(
            "agenticrag.api:app", host="127.0.0.1", port=9000, reload=True
        )


def test_cli_default_command_runs_serve():
    """When no subcommand is supplied, default to serve."""
    with (
        patch("sys.argv", ["agenticrag"]),
        patch("uvicorn.run") as mock_uvicorn_run,
    ):
        main()
        mock_uvicorn_run.assert_called_once_with(
            "agenticrag.api:app", host="0.0.0.0", port=8000, reload=False
        )


def test_cli_ingest_invokes_run_ingestion():
    """Test that ingest subcommand calls run_ingestion."""
    with (
        patch("sys.argv", ["agenticrag", "ingest"]),
        patch("agenticrag.ingest.run_ingestion") as mock_run_ingest,
    ):
        main()
        mock_run_ingest.assert_called_once()


def test_cli_ingest_missing_datasets_handles_gracefully(capsys):
    """Test that missing datasets library displays installation instructions and exits with code 1."""
    import builtins

    orig_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "agenticrag.ingest":
            raise ImportError("No module named 'datasets'")
        return orig_import(name, *args, **kwargs)

    with (
        patch("sys.argv", ["agenticrag", "ingest"]),
        patch("builtins.__import__", side_effect=fake_import),
    ):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "The 'datasets' library is required" in captured.err
