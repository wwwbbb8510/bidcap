import setuptools
import os

with open(os.path.join('bidcap', "README.md"), "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bidcap",
    version="0.0.1",
    author="Bin Wang",
    author_email="wwwbbb8510@gmail.com",
    description="benchmark image dataset collection and preprocessing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wwwbbb8510/bidcap.git",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)