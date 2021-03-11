
# -*- coding: utf-8 -*-

# This is a shim setup.py file which only serves the purpose of allowing us
# to use setuptools to create an editable install during development,
# i.e. it allows us to run `pip install --editable .` which will
# create a symbolic link from your environment's `site-packages` directory
# to the Lightkurve source code tree. Note that this is NOT the recommended
# way to develop lightkurve; we recommend using the `poetry` build tool instead.
# For more information, see https://docs.lightkurve.org/about/contributing.html
# and https://snarky.ca/what-the-heck-is-pyproject-toml/

# DO NOT EDIT THIS FILE!
# This file has been autogenerated by dephell <3
# https://github.com/dephell/dephell

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


import os.path

readme = ''
here = os.path.abspath(os.path.dirname(__file__))
readme_path = os.path.join(here, 'README.rst')
if os.path.exists(readme_path):
    with open(readme_path, 'rb') as stream:
        readme = stream.read().decode('utf8')


setup(
    long_description=readme,
    name='lightkurve',
    version='2.0.5dev',
    description='A friendly package for Kepler & TESS time series analysis in Python.',
    python_requires='==3.*,>=3.6.1',
    project_urls={"homepage": "https://docs.lightkurve.org", "repository": "https://github.com/lightkurve/lightkurve"},
    author='Geert Barentsen',
    author_email='hello@geert.io',
    license='MIT',
    keywords='NASA Kepler TESS Astronomy',
    classifiers=['Intended Audience :: Science/Research', 'Topic :: Scientific/Engineering :: Astronomy', 'Development Status :: 5 - Production/Stable', 'License :: OSI Approved :: MIT License', 'Operating System :: OS Independent', 'Programming Language :: Python'],
    packages=['lightkurve', 'lightkurve.correctors', 'lightkurve.io', 'lightkurve.prf', 'lightkurve.seismology'],
    package_dir={"": "src"},
    package_data={"lightkurve": ["data/*.csv", "data/*.html", "data/*.mplstyle", "data/*.txt"]},
    install_requires=['astropy>=4.1', 'astroquery>=0.3.10', 'beautifulsoup4==4.*,>=4.6.0', 'bokeh>=1.0', 'fbpca>=1.0', 'ipython>=6.0.0', 'matplotlib>=1.5.3', 'memoization>=0.3.1', 'numba>=0.53.0rc1.post1; python_version >= "3.6" and python_version < "3.10"', 'numpy>=1.11', 'oktopus==0.*,>=0.1.2', 'pandas==1.*,>=1.1.4', 'patsy>=0.5.0', 'requests==2.*,>=2.25.0', 'scikit-learn>=0.24.0', 'scipy>=0.19.0', 'tqdm>=4.25.0', 'uncertainties==3.*,>=3.1.4'],
    extras_require={"dev": ["black==20.*,>=20.8.0.b1", "flake8==3.*,>=3.8.4", "ghp-import==1.*,>=1.0.1", "isort==5.*,>=5.6.4", "jupyter==1.*,>=1.0.0", "jupyterlab==2.*,>=2.0.0", "mypy==0.*,>=0.790.0", "nbsphinx==0.*,>=0.8.0", "numpydoc==1.*,>=1.1.0", "pydata-sphinx-theme==0.*,>=0.4.1", "pylint==2.*,>=2.6.0", "pytest==6.*,>=6.1.2", "pytest-cov==2.*,>=2.10.1", "pytest-doctestplus==0.*,>=0.8.0", "pytest-remotedata==0.*,>=0.3.2", "pytest-xdist==2.*,>=2.1.0", "sphinx==3.*,>=3.3.1", "sphinx-automodapi==0.*,>=0.13.0", "sphinxcontrib-rawfiles==0.*,>=0.1.1"]},
)
