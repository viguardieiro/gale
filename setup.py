from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="gale-topo",
    version="0.0.3",
    packages=find_packages(),
    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=[
        "gudhi>=3.8.0",
        "numpy>=1.19.5",
        "networkx>=2.6.3",
        "scikit-learn>=0.24.2",
    ],
    # metadata to display on PyPI
    author="Peter Xenopoulos, Gromit Chan, Harish Doraiswamy, Luis Gustavo Nonato, Brian Barr, Claudio Silva",
    author_email="xenopoulos@nyu.edu",
    description="Globally Assessing Local Explanations (GALE) is a method to compare local explaination output using topological data analysis",
    keywords="explainability interpretability topology",
    url="https://github.com/pnxenopoulos/gale",
    project_urls={
        "Issues": "https://github.com/pnxenopoulos/gale/issues",
        "Documentation": "https://github.com/pnxenopoulos/gale/tree/main/docs",
        "GitHub": "https://github.com/pnxenopoulos/gale/",
    },
    classifiers=["License :: OSI Approved :: MIT License"],
)
