import codecs
from setuptools import setup, find_packages
import os


def load_requirements(file_name):
    """read requirements from requirements.txt file"""
    if os.path.exists(file_name):
        with open(file_name, "r") as file:
            return file.read().splitlines()
    else:
        raise FileNotFoundError(f"{file_name} not found!")

# read dependency packages
requirements = load_requirements("requirements.txt")

# these things are needed for the README.md show on pypi
here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.1'
DESCRIPTION = 'A Multi-Modality Benchmarking Platform For Molecular Representation'

# define setup setting
setup(
    name="benchmol",  # package name
    version=VERSION,  # version
    author="Hongxin Xiang",  # name
    author_email="",  # email
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=requirements,  # dependency packages
    keywords=['python', 'pytorch', 'benchmol', 'multi-modality platform', 'molecular representation learning'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.6",  # python version
    url="https://github.com/HongxinXiang/BenchMol",
)
