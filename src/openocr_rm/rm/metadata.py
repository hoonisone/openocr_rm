import warnings
from pathlib import Path
from typing import Any, Optional


_warned_prop_paths: set[str] = set()


def config_metadata(config: Any) -> Optional[dict]:
    if not isinstance(config, dict):
        return None
    meta = config.get("Metadata")
    if isinstance(meta, dict) and meta:
        return meta
    return None


def warn_deprecated_prop(prop_path: Path) -> None:
    key = str(Path(prop_path).resolve())
    if key in _warned_prop_paths:
        return
    _warned_prop_paths.add(key)
    warnings.warn(
        f"prop.py is deprecated ({prop_path}). "
        "Move its contents into config.py as Metadata.",
        UserWarning,
        stacklevel=3,
    )


def resolve_record_metadata(
    config: Any,
    prop_path: Path,
    config_manager,
    *,
    handling: bool = True,
    lazy_handling: bool = False,
) -> Any:
    meta = config_metadata(config)
    if meta is not None:
        return meta

    prop_path = Path(prop_path)
    if prop_path.exists():
        loaded = config_manager.load_config(
            prop_path, handling=handling, lazy_handling=lazy_handling
        )
        warn_deprecated_prop(prop_path)
        return loaded

    raise ValueError(
        f"No Metadata in config and no prop.py at {prop_path}. "
        "Put record metadata in config.py as Metadata."
    )
