from setuptools import setup, find_packages

setup(
    name='vbe',
    version='0.1',
    description='A library for calculating observable voting bloc entropy (VBE) in DAO proposal voting data.',
    author='Amy Zhao',
    packages=find_packages(include=['vbe', 'vbe.*']),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
    ]
)