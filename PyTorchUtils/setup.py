import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytorchutils_v0",
    version="0.0.1",
    author="Keith Rush",
    author_email="j.keith.rush@gmail.com",
    description="Test of packaging PyTorchUtils for use in @jkr26's projects",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jkr26/pytorchutils",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License 2.0",
        "Operating System :: Linux",
    ),
)