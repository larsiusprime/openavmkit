from setuptools import setup, find_packages

setup(
    name='openavmkit',
    version='0.1.0',
    author='Lars A. Doucet',
    author_email='lars.doucet@gmail.com',
    description='Mass Appraisal and Automated Valuation Modeling tools',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/larsiusprime/openavmkit',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
    install_requires=[
        "numpy>=1.26",
        "pandas>=2.2",
        "tabulate>=0.9.0"
    ],
)