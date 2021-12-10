import setuptools

setuptools.setup(
    name="smcl-maxjcohen",
    version="0.0.1",
    author="Max Cohen",
    author_email="lol44zla5@relay.firefox.com",
    description="Sequential Monte Carlo Layer.",
    url="https://github.com/maxjcohen/smcl",
    packages=["smcl"],
    python_requires=">=3.9",
    install_requires=[
        "torch",
    ],
)
