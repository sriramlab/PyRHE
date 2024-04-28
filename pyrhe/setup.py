from setuptools import setup, find_packages

setup(
    name='pyrhe',  
    version='1.0.0',  
    author="Jiayi Ni",
    author_email='nijiayi1119626@g.ucla.edu',
    description='A Python package for Randomized Haseman-Elston regression for Multi-variance Components.',
    long_description_content_type='text/markdown',
    packages=find_packages(where='src'),  
    package_dir={'': 'src'},  
    install_requires=[
        'bed_reader==1.0.2',
        'numpy>=1.23.1',
        'pandas>=1.5.1',
        'python-dotenv==1.0.1',
        'scipy>=1.10.1',
        'tqdm==4.65.0',
        'configparser>=6.0.1'
    ],
    classifiers=[
        'License :: OSI Approved :: MIT License',
    ],
    python_requires='>=3.8',  
)
