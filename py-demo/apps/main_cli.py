from pathlib import Path

import click


@click.group()
@click.version_option("1.0.0", prog_name="mycli", message="%(prog)s v%(version)s")
def cli():
    pass


@cli.command()
@click.option("--name", default="world", help="user name", type=str)
@click.option("--debug", help="debug mode", is_flag=True)
def greet(name: str, debug: bool) -> None:
    if debug:
        click.echo("debug mode On")
        return
    click.echo(f"hello {name}")


@cli.command()
@click.option("--a", type=int)
@click.option("--b", type=int)
def add(a: int, b: int) -> None:
    click.echo(f"{a+b}")


@cli.command()
@click.option("--file", type=click.Path(exists=True, file_okay=True))
def stat(file: str) -> None:
    fstat = Path(file).stat()
    click.echo(f"[{file}] file size: {fstat.st_size} (bytes)")


if __name__ == "__main__":
    # uv run apps/main_cli.py --help
    # uv run apps/main_cli.py --version

    # uv run apps/main_cli.py greet --name Foo
    # uv run apps/main_cli.py greet --debug

    # uv run apps/main_cli.py add --a=3 --b=5

    # uv run apps/main_cli.py stat --file /tmp/test/output.json
    cli()
