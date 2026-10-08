"""Frozen original AST hashes guard unchanged learning math and control flow."""
import ast
import hashlib
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / "tests/fixtures/jaxnav_frozen_functions.json").read_text())


@pytest.mark.parametrize("path", FIXTURE)
def test_frozen_learning_functions(path):
    tree = ast.parse((ROOT/path).read_text())
    actual = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            actual[node.name] = node
        elif isinstance(node, ast.ClassDef):
            for method in node.body:
                if isinstance(method, ast.FunctionDef):
                    actual[node.name + "." + method.name] = method
    for name, expected in FIXTURE[path]["functions"].items():
        assert name in actual, name
        digest = hashlib.sha256(ast.dump(actual[name],include_attributes=False).encode()).hexdigest()
        assert digest == expected, (path,name)
