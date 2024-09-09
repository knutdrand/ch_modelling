#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['typer', 'gluonts', 'plotly', 'pandas', 'matplotlib',
                'gluonts[torch]',
                'git+https://github.com/dhis2/chap-core.git@dev']

test_requirements = ['pytest>=3', "hypothesis"]

setup(
    author="Knut Rand",
    author_email='knutdrand@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Models and modelling utilities for climate and health",
    entry_points={
        'console_scripts': [
            'ch_modelling=ch_modelling.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='ch_modelling',
    name='ch_modelling',
    packages=find_packages(include=['ch_modelling', 'ch_modelling.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/knutdrand/ch_modelling',
    version='0.0.1',
    zip_safe=False,
)
