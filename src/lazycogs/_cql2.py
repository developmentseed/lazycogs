from typing import Any

from cql2 import Expr


def _extract_filter_fields(filter_expr: str | dict[str, Any]) -> set[str]:
    """Extract all property field names from a CQL2 filter expression."""
    properties: set[str] = set()

    def _traverse(node: object) -> None:
        if isinstance(node, dict):
            if "property" in node:
                properties.add(node["property"])
            else:
                for value in node.values():
                    if isinstance(value, (dict, list)):
                        _traverse(value)
        elif isinstance(node, list):
            for item in node:
                _traverse(item)

    _traverse(Expr(filter_expr).to_json())

    return properties


def _sortby_fields(sortby: str | list[str | dict[str, str]] | None) -> set[str]:
    """Extract property field names from a rustac sortby value."""
    if sortby is None:
        return set()

    if isinstance(sortby, str):
        return {sortby.lstrip("+-")}

    fields: set[str] = set()
    for item in sortby:
        if isinstance(item, str):
            fields.add(item.lstrip("+-"))
        elif isinstance(item, dict):
            name = item.get("field", "")
            if name:
                fields.add(name)

    return fields
