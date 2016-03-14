from setuptools import setup, find_packages
setup(
    name = "ILMpy",
    version = "0.1",
    packages = find_packages(),
    scripts = ['bin/ilm'],

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires = ['docutils>=0.3','pandas','enum'],

    package_data = {
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.txt', '*.rst', '*.pdf'],
        # And include any *.msg files found in the 'hello' package, too:
        'hello': ['*.msg'],
    },

    entry_points = {
        'console_scripts': [
            'ilm = ilmpy.__main__:main',
        ]
    },
    
    # metadata for upload to PyPI
    author = "David H. Ardell",
    author_email = "dardell@ucmerced.edu",
    description = 'Iterated Learning Models in Python',
    license = "Artistic 2.0",
    keywords = "",
    url = "http://pypi.python.org/pypi/ILMpy/",
    long_description=open('README.txt').read(),
)
