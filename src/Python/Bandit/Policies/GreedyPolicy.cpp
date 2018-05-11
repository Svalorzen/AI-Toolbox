#include <AIToolbox/Bandit/Policies/GreedyPolicy.hpp>

#include <boost/python.hpp>

void exportBanditGreedyPolicy() {
    using namespace AIToolbox;
    using namespace boost::python;

    class_<Bandit::GreedyPolicy, bases<Bandit::PolicyInterface>>{"GreedyPolicy",

         "This class models a simple greedy policy.\n"
         "\n"
         "This class always selects the greediest action with respect to the\n"
         "already obtained experience.", no_init}

        .def(init<size_t>(
                 "Basic constructor.\n"
                 "\n"
                 "@param A The size of the action space."
        , (arg("self"), "A")))

        .def("stepUpdateP",             &Bandit::GreedyPolicy::stepUpdateP,
                 "This function updates the greedy policy based on the result of the action.\n"
                 "\n"
                 "We simply keep a rolling average for each action, which we\n"
                 "update here. The ones with the best average are the ones which\n"
                 "will be selected when sampling.\n"
                 "\n"
                 "@param a The action taken.\n"
                 "@param r The reward obtained."
        , (arg("self"), "a", "r"));
}
