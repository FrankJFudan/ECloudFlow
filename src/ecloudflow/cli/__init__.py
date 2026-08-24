"""Typer commands for the ECloudFlow application services.

The application is imported lazily so ``python -m ecloudflow.cli.main`` does
not load the target module before ``runpy`` executes it.
"""

__all__ = ["app"]


def __getattr__(name: str):
    """Resolve the public Typer application without an eager module import."""
    if name == "app":
        from ecloudflow.cli.main import app

        return app
    raise AttributeError(name)
