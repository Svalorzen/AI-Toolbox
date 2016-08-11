#include <AIToolbox/MDP/Policies/QGreedyPolicy.hpp>

#include <boost/python.hpp>

void exportMDPQGreedyPolicy() {
    using namespace AIToolbox::MDP;
    using namespace boost::python;

    class_<QGreedyPolicy, bases<QPolicyInterface>>{"QGreedyPolicy",

         "This class models a greedy policy through a QFunction.\n"
         "\n"
         "This class allows you to select effortlessly the best greedy actions\n"
         "from a given QFunction.", no_init}

        .def(init<const QFunction &>(
                 "Basic constructor.\n"
                 "\n"
                 "@param q The QFunction this policy is linked with."
        , (arg("self"), "q")));
}
