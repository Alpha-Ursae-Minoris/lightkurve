from setuptools import setup, find_packages

setup(
    name="lightkurve",
    version="2.5.0",
    description="A friendly package for Kepler & TESS time series analysis in Python.",
    long_description=open("README.rst").read(),  # Ensure README.rst exists
    license="MIT",
    author="Geert Barentsen",
    author_email="hello@geert.io",
    url="https://github.com/lightkurve/lightkurve",
    homepage="https://docs.lightkurve.org",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    keywords=["NASA", "Kepler", "TESS", "Astronomy"],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.18",
        "astropy>=5.0",
        "scipy>=1.7; python_version >= '3.8' and python_version < '3.11'",
        "matplotlib>=3.1",
        "astroquery>=0.3.10",
        "oktopus>=0.1.2",
        "beautifulsoup4>=4.6.0",
        "requests>=2.22.0",
        "urllib3>=1.23; python_version >= '3.8' and python_version < '4.0'",
        "tqdm>=4.25.0",
        "pandas>=1.1.4",
        "uncertainties>=3.1.4",
        "patsy>=0.5.0",
        "fbpca>=1.0",
        "bokeh>=2.3.2",
        "memoization>=0.3.1; python_version >= '3.8' and python_version < '4.0'",
        "scikit-learn>=0.24.0",
        "s3fs>=2024.6.1",
    ],
    extras_require={
        "dev": [
            "jupyterlab>=2.0.0",
            "black>=21.12b0",
            "flake8>=3.8.4",
            "mypy>=0.930",
            "isort>=5.6.4; python_version >= '3.6' and python_version < '4.0'",
            "pytest>=6.1.2",
            "pytest-cov>=2.10.1",
            "pytest-remotedata>=0.3.2",
            "pytest-doctestplus>=0.8.0",
            "pytest-xdist>=2.1.0",
            "jupyter>=1.0.0",
            "Sphinx>=4.3.0",
            "nbsphinx>=0.8.7",
            "numpydoc>=1.1.0",
            "sphinx-automodapi>=0.13",
            "sphinxcontrib-rawfiles>=0.1.1",
            "pydata-sphinx-theme==0.8.1",
            "pylint>=2.6.0",
            "ghp-import>=1.0.1",
            "openpyxl>=3.0.7",
            "tox>=3.24.5",
            "mistune<2.0.0",  # Workaround for #1162
            "docutils!=0.21",  # Exclude problematic version
        ]
    },
    test_suite="tests",
    tests_require=[
        "pytest>=6.1.2",
    ],
)
