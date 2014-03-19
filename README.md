AI-Toolbox
==========

This C++ toolbox is aimed at representing and solving common AI problems,
implementing an easy-to-use interface which should be hopefully extensible
to many problems, while keeping code readable.

Current development includes MDPs, POMDPs and related algorithms. This toolbox
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
A number of small tests are included which you can find in the `test/` folder.
You can execute them after building the project using the following command
in the build directory:

    ctest

The tests also offer a brief introduction for the framework, waiting for a
more complete descriptive write-up.

To compile the library's documentation you need the [Doxygen](http://www.stack.nl/~dimitri/doxygen/)
tool. To use it it is sufficient to execute the following command from the
project's main folder:

    doxygen

After that the documentation will be generated into an `html` folder in the
main directory.

Documentation
=============

The latest documentation is available [here](http://svalorzen.github.io/AI-Toolbox/).
Keep in mind that it may not always be 100% up to date with the latest
commits, while the one you compile yourself will of course be.
