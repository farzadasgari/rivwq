import numpy as np
import pandas as pd


def tidy_series(
    times,
    values=None,
    code="63680",
    sensor="s1",
    site="0001",
    qualifiers="",
    status="Approved",
):
    index = pd.DatetimeIndex(times)
    if index.tz is None:
        index = index.tz_localize("UTC")
    count = len(index)
    return pd.DataFrame(
        {
            "site_no": site,
            "datetime_utc": index,
            "parameter_code": code,
            "sensor_id": sensor,
            "value": np.arange(count, dtype=float) if values is None else np.asarray(values, float),
            "qualifiers": qualifiers,
            "approval_status": status,
            "unit_of_measure": "u",
            "source_column": "src",
        }
    )


def regular(start, periods, freq, **kwargs):
    times = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    return tidy_series(times, **kwargs)


def scene_times(*stamps):
    return pd.Series(pd.to_datetime(list(stamps), utc=True))
