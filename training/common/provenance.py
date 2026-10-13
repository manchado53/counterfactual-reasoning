"""Recoverable Python sources from either a checkout or an installed wheel."""
from importlib.util import find_spec
from pathlib import Path


def project_sources():
    files = {}
    for package, prefix in (("counterfactual_rl", "src/counterfactual_rl"),
                            ("training", "training")):
        spec = find_spec(package)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"Missing installed package: {package}")
        root = Path(spec.origin).resolve().parent
        for file in sorted(root.rglob("*.py")):
            files[prefix + "/" + file.relative_to(root).as_posix()] = file
    return files
