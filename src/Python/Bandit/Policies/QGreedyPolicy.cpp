#include <AIToolbox/Bandit/Policies/QGreedyPolicy.hpp>

#include <boost/python.hpp>

void exportBanditQGreedyPolicy() {
    using namespace AIToolbox::Bandit;
    using namespace boost::python;

    class_<QGreedyPolicy, bases<PolicyInterface>>{"QGreedyPolicy",

         "This class models a simple greedy policy.\n"
         "\n"
         "This class always selects the greediest action with respect to the\n"
         "already obtained experience.", no_init}

        .def(init<const QFunction &>(
                 "Basic constructor.\n"
                 "\n"
                 "@param q The QFunction to use."
        , (arg("self"), "q")));
}
