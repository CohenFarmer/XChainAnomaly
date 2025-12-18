from datetime import datetime, timezone


def normalize_to_day_start_iso(ts_str: str) -> str:
    if not ts_str:
        return ts_str
    dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).astimezone(timezone.utc)
    normalized = datetime(dt.year, dt.month, dt.day, 0, 0, 0, tzinfo=timezone.utc)
    return normalized.isoformat().replace("+00:00", "Z")