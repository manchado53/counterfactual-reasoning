"""
Setup configuration for counterfactual-reasoning package.

This makes the package installable via pip:
    pip install -e .
    
Or from a URL:
    pip install git+https://github.com/username/counterfactual-reasoning.git
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Dependencies - also specified in pyproject.toml for modern builds
install_requires = [
    "numpy>=1.20.0",
    "gymnasium>=0.27.0",
    "stable-baselines3>=1.6.0",
    "matplotlib>=3.5.0",
    "scipy>=1.7.0",
    "SMAC>=1.0.0",  # StarCraft Multi-Agent Challenge
    "pysc2>=3.0.0",  # StarCraft II API
]

setup(
    name="counterfactual-reasoning",
    version="0.1.0",
    author="RL Research Team",
    description="A framework for analyzing consequential states in RL using counterfactual reasoning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/counterfactual-reasoning",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=3.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            # Add CLI commands here if needed
        ],
    },
)
