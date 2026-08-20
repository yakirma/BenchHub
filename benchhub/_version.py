"""Single source of truth for the benchhub-client version.

`pyproject.toml` reads this via `[tool.setuptools.dynamic]` (so the built
package version follows it), and the server's submission-script generator
reads it to pin the client version it emits. Bump this one line on a client
release — nothing else hardcodes the version.
"""

__version__ = "0.1.11"
