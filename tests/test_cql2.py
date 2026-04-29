"""Tests for the lazycogs._cql2 module."""

from __future__ import annotations

from lazycogs._cql2 import _extract_filter_fields, _sortby_fields

# ---------------------------------------------------------------------------
# _extract_filter_fields
# ---------------------------------------------------------------------------


def test_extract_filter_fields_single_property():
    """A simple comparison extracts one property name."""
    fields = _extract_filter_fields("eo:cloud_cover < 10")
    assert fields == {"eo:cloud_cover"}


def test_extract_filter_fields_multiple_properties():
    """An AND expression extracts all referenced property names."""
    fields = _extract_filter_fields("eo:cloud_cover < 10 AND platform = 'sentinel-2a'")
    assert fields == {"eo:cloud_cover", "platform"}


def test_extract_filter_fields_dict_input():
    """A CQL2-JSON dict is accepted directly."""
    cql2_json = {
        "op": "lt",
        "args": [{"property": "eo:cloud_cover"}, 20],
    }
    fields = _extract_filter_fields(cql2_json)
    assert fields == {"eo:cloud_cover"}


def test_extract_filter_fields_returns_unique():
    """The same property referenced twice appears only once."""
    fields = _extract_filter_fields("eo:cloud_cover > 0 AND eo:cloud_cover < 50")
    assert fields == {"eo:cloud_cover"}


def test_extract_filter_fields_nested():
    """Properties nested inside OR expressions are found recursively."""
    fields = _extract_filter_fields(
        "(eo:cloud_cover < 10 OR eo:cloud_cover > 90) AND platform = 'landsat-8'",
    )
    assert fields == {"eo:cloud_cover", "platform"}


# ---------------------------------------------------------------------------
# _sortby_fields
# ---------------------------------------------------------------------------


def test_sortby_fields_none():
    """None returns an empty set."""
    assert _sortby_fields(None) == set()


def test_sortby_fields_bare_string():
    """A bare field name returns a one-element set."""
    assert _sortby_fields("datetime") == {"datetime"}


def test_sortby_fields_prefixed_string_plus():
    """A '+'-prefixed string has the prefix stripped."""
    assert _sortby_fields("+datetime") == {"datetime"}


def test_sortby_fields_prefixed_string_minus():
    """A '-'-prefixed string has the prefix stripped."""
    assert _sortby_fields("-datetime") == {"datetime"}


def test_sortby_fields_list_of_strings():
    """A list of prefixed strings returns all bare names."""
    assert _sortby_fields(["+datetime", "-eo:cloud_cover"]) == {
        "datetime",
        "eo:cloud_cover",
    }


def test_sortby_fields_list_of_dicts():
    """A list of field/direction dicts returns all field names."""
    sortby = [
        {"field": "datetime", "direction": "asc"},
        {"field": "eo:cloud_cover", "direction": "desc"},
    ]
    assert _sortby_fields(sortby) == {"datetime", "eo:cloud_cover"}


def test_sortby_fields_mixed_list():
    """A mixed list of strings and dicts is handled correctly."""
    sortby = ["+datetime", {"field": "platform", "direction": "asc"}]
    assert _sortby_fields(sortby) == {"datetime", "platform"}


def test_sortby_fields_dict_missing_field_key():
    """A dict without a 'field' key is silently skipped."""
    assert _sortby_fields([{"direction": "asc"}]) == set()


def test_sortby_fields_deduplicated():
    """Duplicate field names appear only once."""
    assert _sortby_fields(["+datetime", "-datetime"]) == {"datetime"}
