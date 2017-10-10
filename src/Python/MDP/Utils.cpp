#include <AIToolbox/MDP/Utils.hpp>

#include <boost/python.hpp>

void exportMDPUtils() {
    using namespace boost::python;
    using namespace AIToolbox::MDP;

    def("makeQFunction", makeQFunction);
}
