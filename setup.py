import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="self_organising_systems",
    version="0.1",
    author="Alexander Mordvintsev, Ettore Randazzo, Eyvind Niklasson",
    author_email="moralex@google.com, etr@google.com, eyvind@google.com",
    description="Code relating to selforg work.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/google-research/self-organizing-systems",
    packages=setuptools.find_packages(),
    install_requires=[
        "evojax",  # biomakerca dependency
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.0',
)
