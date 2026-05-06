"""MkDocs hook that reads package versions from pyproject.toml files
and injects them into config.extra so they are available in Jinja2 templates."""

import tomllib
from pathlib import Path


def on_config(config):
    root = Path(config["config_file_path"]).parent

    with open(root / "pyproject.toml", "rb") as f:
        ext_version = tomllib.load(f)["project"]["version"]

    with open(root / "jumper_wrapper_kernel" / "pyproject.toml", "rb") as f:
        wrapper_version = tomllib.load(f)["project"]["version"]

    config["extra"]["ext_version"] = ext_version
    config["extra"]["wrapper_version"] = wrapper_version

    return config
