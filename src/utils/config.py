"""Run specs: a flat TOML file whose keys are train.py's CLI options."""

import tomllib
from pathlib import Path


def apply_config_file(parser, path):
    """Sets parser defaults from a TOML file, so explicit CLI flags still override it.

    Raises ValueError for keys that aren't options of the parser.
    """
    values = tomllib.loads(Path(path).read_text())
    known = {a.dest for a in parser._actions}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"{path}: unknown keys {unknown}")
    parser.set_defaults(**values)
