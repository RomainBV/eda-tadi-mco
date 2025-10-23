import os
import sys

from setuptools import find_namespace_packages, setup
from setuptools.command.test import test as test_command

if os.path.exists("README.md"):
    long_description = open("README.md").read()
else:
    long_description = "A Python library for signal processing"


class PyTest(test_command):
    def finalize_options(self):
        test_command.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest

        err_no = pytest.main(self.test_args)
        sys.exit(err_no)


extras_require = {
    "test": [
        "coverage>=4.5.1,<5.0.0",
        "isort>=4.3.4,<5.0.0",
        "mccabe>=0.6.1,<1.0.0",
        "pyflakes>=2.0.0,<3.0.0",
        "pylama>=7.4.3,<8.0.0",
        "pytest",
        "soundfile",
        "tox>=3.0.0,<4.0.0",
        "tqdm",
    ],
    "dev": [
        "alabaster>=0.7.10,<1.0.0",
        "bump2version",
        "pre-commit",
        "Pygments>=2.2.0,<3.0.0",
        "sphinx-autodoc-typehints>=1.6.0,<2.0.0",
        "sphinx-rtd-theme>=0.4.3,<1.0.0",
        "Sphinx>=1.7.5,<2.0.0",
        "wheel",
    ],
}

extras_require["notebook"] = [
    "boto3",
    "dvc[s3]<1.0.0",
    "flufl.lock>=3.2,<4.0",
    "jupyter",
    "matplotlib",
    "nb_black",
    "tables>=3.0.0,<4.0.0",
    "pandas>=0.25.3,<2.0.0",
]

extras_require["test"] = extras_require["test"] + extras_require["notebook"]

extras_require["dev"] = extras_require["dev"] + extras_require["test"]

about = {}
with open(
    os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "wavely", "signal", "__version__.py"
    ),
) as f:
    exec(f.read(), about)

setup(
    name="wavely.signal",
    version=about["__version__"],
    description="Signal processing module for Python.",
    long_description_content_type="text/markdown",
    long_description=long_description,
    author="Wavely",
    author_email="contact@wavely.fr",
    url="https://github.com/Wavely/signal",
    packages=find_namespace_packages(include=["wavely.*"]),
    install_requires=[
        "dynaconf[yaml]>=2.2.2,<3.0.0",
        'numpy>=1.14.2,<1.20.0; python_version == "3.5"',
        'numpy>=1.14.2,<2.0.0; python_version > "3.5"',
        "scipy>=1.0.1,<1.14.0",
    ],
    extras_require=extras_require,
    include_package_data=True,
    package_data={"wavely.signal": ["py.typed"]},
    python_requires=">=3.5, <4",
)
