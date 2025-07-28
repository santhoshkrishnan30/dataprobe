"""
Setup configuration for DataProbe package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="dataprobe",
    version="2.1.0",
    author="SANTHOSH KRISHNAN R",
    author_email="santhoshkrishnan3006@gmail.com",
    description="Advanced data pipeline debugging and profiling tools for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/santhoshkrishnan30/dataprobe",
    project_urls={
        "Bug Tracker": "https://github.com/santhoshkrishnan30/dataprobe/issues",
        "Documentation": "https://dataprobe.readthedocs.io",
        "Source Code": "https://github.com/santhoshkrishnan30/dataprobe",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Debuggers",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "polars>=0.19.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "rich>=13.0.0",
        "click>=8.0.0",
        "psutil>=5.9.0",
        "memory-profiler>=0.60.0",
        "graphviz>=0.20.0",
        "networkx>=2.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.990",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "all": [
            "sqlalchemy>=2.0.0",
            "pyarrow>=10.0.0",
            "jupyter>=1.0.0",
            "ipython>=8.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "dataprobe=dataprobe.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "data-pipeline",
        "debugging",
        "profiling",
        "data-engineering",
        "etl",
        "data-lineage",
        "memory-profiling",
        "pandas",
        "polars"
    ],
)
