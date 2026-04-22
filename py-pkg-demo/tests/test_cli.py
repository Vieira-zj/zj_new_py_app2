from click.testing import CliRunner

from py_pkg_demo import cli


def test_cli_ls():
    runner = CliRunner()
    result = runner.invoke(cli=cli, args=["ls", "--verbose", "/tmp/test"])
    assert result.exit_code == 0
    print("\noutput:\n" + result.output)


def test_cli_size():
    runner = CliRunner()
    result = runner.invoke(cli=cli, args=["size", "/tmp/test/output.json"])
    assert result.exit_code == 0
    print("\noutput:\n" + result.output)


def test_cli_find():
    runner = CliRunner()
    result = runner.invoke(
        cli=cli, args=["find", "-v", "--key=tool", "~/Downloads/tmps"]
    )
    assert result.exit_code == 0
    print("\noutput:\n" + result.output)
