AI-Toolbox
==========

This C++ toolbox is aimed at representing and solving common AI problems,
implementing an easy-to-use interface which should be hopefully extensible
to many problems, while keeping code readable.

Current development includes MDPs and related algorithms. This toolbox
has been developed taking inspiration from the Matlab MDPToolbox,
which you can find [here](http://www7.inra.fr/mia/T/MDPtoolbox/).

Build Instructions
==================

To build the library you need to install [cmake](http://www.cmake.org/) and
the [boost library](http://www.boost.org/) >= 1.53.

After that, you can simply execute the following commands from the project's
main folder:

    mkdir build
    cd build/
    cmake ..
    make

The static library file will be available directly in the `build` directory.
In addition, a small program will be outputted in the project's main folder.
The program's source is in the `test` folder and offers a small example of
what the library can do.

To compile the library's documentation you need the [Doxygen](http://www.stack.nl/~dimitri/doxygen/)
tool. To use it it is sufficient to execute the following command from the
project's main folder:

    doxygen

After that the documentation will be generated into an `html` folder in the
main directory.
