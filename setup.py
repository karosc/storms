#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

try:  # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError:  # for pip <= 9.0.3
    from pip.req import parse_requirements

with open("README.md") as readme_file:
    readme = readme_file.read()

# with open("HISTORY.rst") as history_file:
#     history = history_file.read()

# # parse_requirements() returns generator of pip.req.InstallRequirement objects
# install_reqs = parse_requirements("requirements.txt", session=False)
# # reqs is a list of requirement
# try:
#     requirements = [str(ir.req) for ir in install_reqs]
# except:
#     requirements = [str(ir.requirement) for ir in install_reqs]

test_requirements = [
    "pytest>=3",
]

requirements = [
    "numpy",
    "pandas",
    "numba",
    "scipy",
    "jinja2",
    "requests",
    "aiohttp",
    "aiohttp-retry",
    "tqdm",
    "shapely",
    "pyproj",
    "Pillow",
    "lmoments3@git+https://github.com/OpenHydrology/lmoments3.git@7e19f97c23019ca68cbd526b8bd417c412438f1c",
]

setup(
    author="Constantine Karos",
    author_email="ckaros@outlook.com",
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",

    ],
    description="Python water data and statistics library",
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="storms",
    name="storms",
    packages=find_packages(include=["storms", "storms.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/karosc/storms",
    version="0.1.0",
    zip_safe=False,
)
