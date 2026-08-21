"""Sample recipes under examples/recipes/ must load and pass schema checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from cloudet.reduction.session import _check_recipe, load_recipe

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "recipes"

_SAMPLE_FILES = (
    "tracker_planes.json",
    "marker_baseline.json",
    "duct_on_wall.json",
)


@pytest.mark.parametrize("name", _SAMPLE_FILES)
def test_example_recipe_loads_and_checks(name: str):
    path = _EXAMPLES / name
    assert path.is_file(), f"missing sample recipe {path}"
    recipe = load_recipe(path)
    _check_recipe(recipe)
    assert recipe["version"] == 2
    assert recipe["units"] == "mm"
    assert recipe["faces"]
