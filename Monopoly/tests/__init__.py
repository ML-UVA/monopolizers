"""Package marker for Monopoly subpackage tests.

This allows tests under `monopolizers.Monopoly.tests` to be imported
as package modules so relative imports (e.g. `from ..state import ...`)
work during pytest collection.
"""

__all__ = []
