Installation
============================================
It is recommended to install all dependencies and run skILMpy with uv. 
Instructions for downloading uv can be found here: https://docs.astral.sh/uv/
After uv is installed, and this repository has been cloned to your system
set your working directory accordingly. 

In the directory for skILMpy on your system run "uv sync", in order to install
all the required dependencies. Followed by "uv run ilm.py" to run the program.
Any commands must have uv run before the ilm.py script


Dependencies 
============================================ 

relies heavily on, and absolutely requires, numpy as a prerequisite.
You should install numpy with the easy_install framework to be detected as
installed when installing this package. 

numpy
pandas,ply,distance,sympy

Usage
============================================

ILMpy comes with an executable inside the bin subdirectory to the
installation source package, a UNIX-compatible script called "ilm.py". 

Additionally, a platform-specific executable may be automatically generated
on installation.

Also try running the --help option to the executables after installation and
for a command-line example.

Programmers may use the executable in bin as a guide and template for how to
program against the cmcpy API.
		       			  
Documentation 
============================================ 

Some documentation of the cmcpy API is available within the "doc"
subdirectory of the source distribution. HTML, pdf and texinfo alternative
formats are provided.

Licensing and Attribution 
============================================



Release Notes
============================================


See CHANGES.txt for version-related changes.

References
============================================
