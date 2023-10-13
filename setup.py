from setuptools import setup, find_packages

setup(
    name='RHE',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
    ],
    extras_require={
        'dev': [
            # List additional groups of dependencies here (e.g. development dependencies).
            # 'pytest',
            # 'flake8',
        ],
    },
    python_requires='>=3.6, <4',
    entry_points={
        'console_scripts': [
        ],
    },
)
