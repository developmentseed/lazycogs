"""Private temporal grouping logic for STAC item time-step bucketing."""

from __future__ import annotations

import calendar
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import numpy as np

# Epoch used for epoch-aligned fixed-duration periods.
_EPOCH = date(2000, 1, 1)
_DATETIME_EPOCH = datetime(2000, 1, 1, tzinfo=UTC)

_ISO_DURATION_RE = re.compile(r"^P(\d+)(D|W|M|Y)$")
_ISO_HOUR_DURATION_RE = re.compile(r"^PT(\d+)H$")


@dataclass(frozen=True)
class _TimeStep:
    """Internal temporal step with coordinate and rustac datetime predicate.

    Attributes:
        coord: The xarray coordinate value for this time step.
        label: Opaque sortable grouping label.
        datetime_filter: Predicate passed to ``rustac`` as ``datetime=``.

    """

    coord: np.datetime64
    label: str
    datetime_filter: str


class _TemporalGrouper(ABC):
    """Abstract base for temporal grouping strategies.

    Each subclass buckets STAC item datetimes into discrete time steps,
    producing a group label (used for sorting and deduplication), a
    ``rustac``-compatible datetime filter string, and a ``numpy.datetime64``
    coordinate value.

    """

    @abstractmethod
    def group_key(self, datetime_str: str) -> str:
        """Map a STAC item datetime string to a sortable group label."""
        ...

    @abstractmethod
    def datetime_filter(self, group_key: str) -> str:
        """Return a ``rustac``-compatible datetime filter for a group."""
        ...

    @abstractmethod
    def to_datetime64(self, group_key: str) -> np.datetime64:
        """Map a group label to an xarray time coordinate value."""
        ...

    def time_step(self, group_key: str) -> _TimeStep:
        """Build a complete time-step object from a group key."""
        return _TimeStep(
            coord=self.to_datetime64(group_key),
            label=group_key,
            datetime_filter=self.datetime_filter(group_key),
        )


def _parse_timestamp(datetime_str: str) -> datetime:
    """Parse a timestamp and normalize it to UTC.

    Bare dates are rejected because exact and sub-daily grouping need a real
    instant rather than rustac's date-wide interpretation of ``YYYY-MM-DD``.
    Naive timestamps are treated as UTC for compatibility with STAC-like test
    fixtures that omit the offset.
    """
    if "T" not in datetime_str:
        raise ValueError(
            f"Timestamp {datetime_str!r} must include a time component for "
            "exact or sub-daily temporal grouping.",
        )
    try:
        parsed = datetime.fromisoformat(datetime_str)
    except ValueError as exc:
        raise ValueError(f"Invalid timestamp {datetime_str!r}.") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _format_utc_timestamp(value: datetime, *, timespec: str = "auto") -> str:
    """Return a stable UTC timestamp string ending in ``Z``."""
    return value.astimezone(UTC).isoformat(timespec=timespec).replace("+00:00", "Z")


class _ExactTimestampGrouper(_TemporalGrouper):
    """Group items by their unique normalized timestamp (``time_period=None``)."""

    def group_key(self, datetime_str: str) -> str:
        """Return the normalized UTC timestamp for *datetime_str*."""
        return _format_utc_timestamp(_parse_timestamp(datetime_str))

    def datetime_filter(self, group_key: str) -> str:
        """Return the normalized timestamp unchanged for rustac."""
        return group_key

    def to_datetime64(self, group_key: str) -> np.datetime64:
        """Return the exact timestamp as ``datetime64[ns]``."""
        return np.datetime64(group_key.removesuffix("Z"), "ns")


class _HourGrouper(_TemporalGrouper):
    """Group items into fixed-length hour windows aligned to the UTC epoch."""

    def __init__(self, n_hours: int) -> None:
        """Initialise with a positive fixed hour window length."""
        if n_hours < 1:
            raise ValueError("Hour temporal grouping requires a positive hour count.")
        self._n = n_hours

    def _bucket_start(self, group_key: str) -> datetime:
        """Return the UTC bucket start represented by *group_key*."""
        return _parse_timestamp(group_key)

    def group_key(self, datetime_str: str) -> str:
        """Return the bucket start timestamp label for *datetime_str*."""
        value = _parse_timestamp(datetime_str)
        bucket_seconds = self._n * 60 * 60
        elapsed = int((value - _DATETIME_EPOCH).total_seconds())
        bucket_start_seconds = (elapsed // bucket_seconds) * bucket_seconds
        start = _DATETIME_EPOCH + timedelta(seconds=bucket_start_seconds)
        return _format_utc_timestamp(start, timespec="seconds")

    def datetime_filter(self, group_key: str) -> str:
        """Return a closed second-precision ``start/end`` range."""
        start = self._bucket_start(group_key)
        end = start + timedelta(hours=self._n, seconds=-1)
        return (
            f"{_format_utc_timestamp(start, timespec='seconds')}/"
            f"{_format_utc_timestamp(end, timespec='seconds')}"
        )

    def to_datetime64(self, group_key: str) -> np.datetime64:
        """Return the bucket start as ``datetime64[s]``."""
        return np.datetime64(group_key.removesuffix("Z"), "s")


class _DayGrouper(_TemporalGrouper):
    """Group items by calendar day (``P1D``)."""

    def group_key(self, datetime_str: str) -> str:
        """Return the ``YYYY-MM-DD`` portion of *datetime_str*."""
        return datetime_str[:10]

    def datetime_filter(self, group_key: str) -> str:
        """Return *group_key* unchanged; rustac accepts bare date strings."""
        return group_key

    def to_datetime64(self, group_key: str) -> np.datetime64:
        """Return ``numpy.datetime64(group_key, "D")``."""
        return np.datetime64(group_key, "D")


class _WeekGrouper(_TemporalGrouper):
    """Group items by ISO 8601 calendar week (``P1W``), anchored on Monday."""

    def group_key(self, datetime_str: str) -> str:
        """Return an ``YYYY-Www`` ISO week label for *datetime_str*."""
        d = date.fromisoformat(datetime_str[:10])
        iso = d.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"

    def datetime_filter(self, group_key: str) -> str:
        """Return a ``Monday/Sunday`` RFC 3339 range for *group_key*."""
        monday = self._monday(group_key)
        sunday = monday + timedelta(days=6)
        return f"{monday.isoformat()}/{sunday.isoformat()}"

    def to_datetime64(self, group_key: str) -> np.datetime64:
        """Return the Monday of the ISO week as ``datetime64[D]``."""
        return np.datetime64(self._monday(group_key).isoformat(), "D")

    @staticmethod
    def _monday(group_key: str) -> date:
        """Return the Monday ``date`` for an ``YYYY-Www`` key."""
        year = int(group_key[:4])
        week = int(group_key[6:])
        jan4 = date(year, 1, 4)
        week1_monday = jan4 - timedelta(days=jan4.weekday())
        return week1_monday + timedelta(weeks=week - 1)


class _MonthGrouper(_TemporalGrouper):
    """Group items by calendar month (``P1M``)."""

    def group_key(self, datetime_str: str) -> str:
        """Return the ``YYYY-MM`` portion of *datetime_str*."""
        return datetime_str[:7]

    def datetime_filter(self, group_key: str) -> str:
        """Return a ``YYYY-MM-01/YYYY-MM-DD`` range covering the full month."""
        year, month = int(group_key[:4]), int(group_key[5:7])
        last_day = calendar.monthrange(year, month)[1]
        return f"{group_key}-01/{group_key}-{last_day:02d}"

    def to_datetime64(self, group_key: str) -> np.datetime64:
        """Return the first of the month as ``datetime64[D]``."""
        return np.datetime64(f"{group_key}-01", "D")


class _YearGrouper(_TemporalGrouper):
    """Group items by calendar year (``P1Y``)."""

    def group_key(self, datetime_str: str) -> str:
        """Return the ``YYYY`` portion of *datetime_str*."""
        return datetime_str[:4]

    def datetime_filter(self, group_key: str) -> str:
        """Return a ``YYYY-01-01/YYYY-12-31`` range covering the full year."""
        return f"{group_key}-01-01/{group_key}-12-31"

    def to_datetime64(self, group_key: str) -> np.datetime64:
        """Return January 1st of the year as ``datetime64[D]``."""
        return np.datetime64(f"{group_key}-01-01", "D")


class _FixedDayGrouper(_TemporalGrouper):
    """Group items into fixed-length windows of *n_days* days."""

    def __init__(self, n_days: int) -> None:
        """Initialise with a fixed window length."""
        self._n = n_days

    def _bucket(self, datetime_str: str) -> int:
        """Return the zero-based bucket index for *datetime_str*."""
        d = date.fromisoformat(datetime_str[:10])
        return (d - _EPOCH).days // self._n

    def group_key(self, datetime_str: str) -> str:
        """Return a zero-padded decimal bucket index as the group label."""
        return f"{self._bucket(datetime_str):06d}"

    def datetime_filter(self, group_key: str) -> str:
        """Return a ``start/end`` RFC 3339 range for the bucket."""
        bucket = int(group_key)
        start = _EPOCH + timedelta(days=bucket * self._n)
        end = start + timedelta(days=self._n - 1)
        return f"{start.isoformat()}/{end.isoformat()}"

    def to_datetime64(self, group_key: str) -> np.datetime64:
        """Return the start date of the bucket as ``datetime64[D]``."""
        bucket = int(group_key)
        start = _EPOCH + timedelta(days=bucket * self._n)
        return np.datetime64(start.isoformat(), "D")


def grouper_from_period(time_period: str | None) -> _TemporalGrouper:
    """Return a temporal grouper for a supported grouping period.

    Supported values are ``None`` for exact timestamps, date durations
    ``P1D``, ``PnD``, ``P1W``, ``P1M``, ``P1Y``, and hour durations ``PTnH``.
    """
    if time_period is None:
        return _ExactTimestampGrouper()

    m = _ISO_DURATION_RE.match(time_period)
    if m:
        count, unit = int(m.group(1)), m.group(2)
        if unit == "D":
            return _DayGrouper() if count == 1 else _FixedDayGrouper(count)
        if unit == "W":
            return _WeekGrouper() if count == 1 else _FixedDayGrouper(count * 7)
        if unit == "M" and count == 1:
            return _MonthGrouper()
        if unit == "Y" and count == 1:
            return _YearGrouper()

    hour_match = _ISO_HOUR_DURATION_RE.match(time_period)
    if hour_match:
        count = int(hour_match.group(1))
        if count > 0:
            return _HourGrouper(count)

    raise ValueError(
        f"Unsupported time_period {time_period!r}. "
        "Supported values: None, 'P1D', 'PnD' (n>1), 'P1W', 'P1M', "
        "'P1Y', and 'PTnH' (n>=1).",
    )
