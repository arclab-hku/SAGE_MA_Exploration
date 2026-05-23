from typing import Iterable, Mapping, Optional


def _uav_sort_key(raw_key: str):
    try:
        return (0, int(raw_key))
    except Exception:
        return (1, str(raw_key))


def sorted_uav_keys(keys: Iterable[str]):
    return sorted((str(k) for k in keys), key=_uav_sort_key)


def uav_key_from_index(mapping: Mapping[str, object], index: int) -> Optional[str]:
    keys = sorted_uav_keys(mapping.keys())
    if index < 0 or index >= len(keys):
        return None
    return keys[index]
