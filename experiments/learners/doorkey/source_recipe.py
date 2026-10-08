"""Load the bundled fixed-transfer recipe from its single JSON owner."""
import json
from importlib.resources import files


RECIPE = json.loads(files(__package__).joinpath("source_recipe.json").read_text())
