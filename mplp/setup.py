import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mplp",
    version="0.1",
    author="Ettore Randazzo, Eyvind Niklasson, Alexander Mordvintsev",
    author_email="etr@google.com, eyvind@google.com, moralex@google.com",
    description="Code relating to MPLP work.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/google-research/self-organizing-systems/mplp",
    packages=["mplp"],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.0',
)
