import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="scarf-matching",
    version="0.0.1",
    author="Dengwang Tang",
    author_email="dwtang@umich.edu",
    description="A package for stable matching problems with couples",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dwtang/scarf/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)