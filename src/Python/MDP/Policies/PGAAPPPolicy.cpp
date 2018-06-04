#include <AIToolbox/MDP/Policies/PGAAPPPolicy.hpp>

#include <boost/python.hpp>

void exportMDPPGAAPPPolicy() {
    using namespace AIToolbox::MDP;
    using namespace boost::python;

    class_<PGAAPPPolicy, bases<QPolicyInterface>>{"PGAAPPPolicy",

         "This class models the WoLF learning algorithm.\n"
         "\n"
         "What this algorithm does is it progressively modifies the policy\n"
         "given changes in the underlying QFunction. In particular, it\n"
         "modifies it rapidly if the agent is 'losing' (getting less reward\n"
         "than expected), and more slowly when 'winning', since there's little\n"
         "reason to change behaviour when things go right.\n"
         "\n"
         "An advantage of this algorithm is that it can allow the policy to\n"
         "converge to non-deterministic solutions: for example two players\n"
         "trying to outmatch each other in rock-paper-scissor. At the same\n"
         "time, this particular version of the algorithm can take quite some\n"
         "time to converge to a good solution.", no_init}

        .def(init<const QFunction &, optional<double,double>>(
                 "Basic constructor.\n"
                 "\n"
                 "See the setter functions to see what the parameters do.\n"
                 "\n"
                 "@param q The QFunction from which to extract policy updates.\n"
                 "@param deltaw The learning rate if this policy is currently winning.\n"
                 "@param deltal The learning rate if this policy is currently losing."
        , (arg("self"), "q", "deltaw", "deltal")))

        .def("stepUpdateP",                 &PGAAPPPolicy::stepUpdateP,
                 "This function updates the WoLF policy based on changes in the QFunction.\n"
                 "\n"
                 "This function should be called between agent's actions,\n"
                 "using the agent's current state.\n"
                 "\n"
                 "@param s The state that needs to be updated."
        , (arg("self"), "s"))

        .def("setLearningRate",             &PGAAPPPolicy::setLearningRate,
                 "This function sets the new learning rate if winning.\n"
                 "\n"
                 "This is the amount that the policy is modified when the updatePolicy() function is called\n"
                 "when WoLFPolicy determines that it is currently winning based on the current QFunction.\n"
                 "\n"
                 "@param deltaW The new learning rate during wins."
        , (arg("self"), "deltaW"))

        .def("getLearningRate",             &PGAAPPPolicy::getLearningRate,
                 "This function returns the current learning rate during winning."
        , (arg("self")))

        .def("setPredictionLength",         &PGAAPPPolicy::setPredictionLength,
                 "This function sets the new learning rate if losing.\n"
                 "\n"
                 "This is the amount that the policy is modified when the updatePolicy() function is called\n"
                 "when WoLFPolicy determines that it is currently losing based on the current QFunction.\n"
                 "\n"
                 "@param deltaL The new learning rate during loss."
        , (arg("self"), "deltaL"))

        .def("getPredictionLength",         &PGAAPPPolicy::getPredictionLength,
                 "This function returns the current learning rate during loss."
        , (arg("self")));
}


