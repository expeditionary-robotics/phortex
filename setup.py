"""Configuration for the fumes package."""
from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fumes",
    version="0.0.1",
    author="Victoria Preston, Genevieve Flaspohler",
    author_email='vpreston@mit.edu, geflaspo@mit.edu',
    description='Package for intermittent mission deployment planner.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    package_dir={"": "fumes"},
    packages=find_packages(where="fumes"),
    install_requires=['numpy',
                      'matplotlib',
                      'scipy',
                      ],
    python_requires=">=3.6",
)
