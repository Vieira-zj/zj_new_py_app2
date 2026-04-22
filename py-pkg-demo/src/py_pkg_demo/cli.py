import click

from .io_file import IOFile


@click.group()
@click.version_option(package_name="py-pkg-demo")
def cli():
    pass


@cli.command()
@click.argument("path", required=True)
@click.option("--verbose", "-v", is_flag=True)
def ls(path: str, verbose: bool):
    if verbose:
        click.echo(f"ls {path}")

    for f in IOFile(path).ls():
        click.echo(f)


@cli.command()
@click.argument("path", required=True)
@click.option("--verbose", "-v", is_flag=True)
def size(path: str, verbose: bool):
    if verbose:
        click.echo(f"size {path}")

    size = IOFile(path).size()
    click.echo(f"{path}: {size}")


@cli.command()
@click.argument("path", required=True)
@click.option("--key", required=True)
@click.option("--verbose", "-v", is_flag=True)
def find(path: str, key: str, verbose: bool):
    if verbose:
        click.echo(f"find --key={key} {path}")

    for f in IOFile(path).find(key):
        click.echo(f)
