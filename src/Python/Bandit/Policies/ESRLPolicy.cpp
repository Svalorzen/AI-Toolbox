#include <AIToolbox/Bandit/Policies/ESRLPolicy.hpp>

#include <boost/python.hpp>

void exportBanditESRLPolicy() {
    using namespace AIToolbox;
    using namespace boost::python;

    class_<Bandit::ESRLPolicy, bases<Bandit::PolicyInterface>>{"ESRLPolicy",

         "This class models the Exploring Selfish Reinforcement Learning algorithm.\n"
         "\n"
         "This is a learning algorithm for common interest games. It tries to\n"
         "consider both Nash equilibria and Pareto-optimal solution in order to\n"
         "maximize the payoffs to the agents.\n"
         "\n"
         "The original algorithm can be modified in order to work with\n"
         "non-cooperative games, but here we implement only the most general\n"
         "version for cooperative games.\n"
         "\n"
         "An important point for this algorithm is that each agent only considers\n"
         "its own payoffs, and in the cooperative case does not need to\n"
         "communicate with the other agents.\n"
         "\n"
         "The idea is to repeatedly use the Linear Reward-Inaction algorithm to\n"
         "converge and find a Nash equilibrium in the space of action, and then\n"
         "cut that one from the action space and repeat the procedure. This would\n"
         "recursively find out all Nash equilibra.\n"
         "\n"
         "This whole process is then repeated multiple times to ensure that most\n"
         "of the equilibria have been explored.\n"
         "\n"
         "During each exploration step, a rolling average is maintained in order\n"
         "to estimate the value of the action the LRI algorithm converged to.\n"
         "\n"
         "After all exploration phases have been done, the best action seen is\n"
         "chosen and repeated forever during the final exploitation phase.", no_init}

        .def(init<size_t, double, unsigned, unsigned, unsigned>(
                 "Basic constructor.\n"
                 "\n"
                 "@param A The size of the action space.\n"
                 "@param a The learning parameter for Linear Reward-Inaction.\n"
                 "@param timesteps The number of timesteps per exploration phase.\n"
                 "@param explorationPhases The number of exploration phases before exploitation.\n"
                 "@param window The last number of timesteps to consider to obtain the learned action value during a single exploration phase."
        , (arg("self"), "A", "a", "timesteps", "explorationPhases", "window")))

        .def("stepUpdateP",             &Bandit::ESRLPolicy::stepUpdateP,
                 "This function updates the ESRL policy based on the result of the action.\n"
                 "\n"
                 "Note that ESRL works with binary rewards: either the action\n"
                 "worked or it didn't.\n"
                 "\n"
                 "Environments where rewards are in R can be simulated: scale all\n"
                 "rewards to the [0,1] range, and stochastically obtain a success\n"
                 "with a probability equal to the reward. The result is equivalent\n"
                 "to the original reward function.\n"
                 "\n"
                 "This function both updates the internal LRI algorithm, and\n"
                 "checks whether a new exploration phase is warranted.\n"
                 "\n"
                 "@param a The action taken.\n"
                 "@param result Whether the action taken was a success, or not."
        , (arg("self"), "a", "result"))

        .def("isExploiting",             &Bandit::ESRLPolicy::isExploiting,
                 "This function returns whether ESRL is now in the exploiting phase.\n"
                 "\n"
                 "This method returns whether ESRLPolicy has finished learning.\n"
                 "Once in the exploiting phase, the method won't learn anymore,\n"
                 "and will simply exploit the knowledge gained.\n"
                 "\n"
                 "Thus, if this method returns true, it won't be necessary anymore\n"
                 "to call the stepUpdateQ method (although it won't have any\n"
                 "effect to do so).\n"
                 "\n"
                 "@return Whether ESRLPolicy is in the exploiting phase."
        , (arg("self")))

        .def("setAParam",                &Bandit::ESRLPolicy::setAParam,
                 "This function sets the a parameter.\n"
                 "\n"
                 "The a parameter determines the amount of learning on successful actions.\n"
                 "\n"
                 "@param a The new a parameter."
        , (arg("self"), "a"))

        .def("getAParam",                &Bandit::ESRLPolicy::getAParam,
                 "This function will return the currently set a parameter.\n"
                 "\n"
                 "@return The currently set a parameter."
        , (arg("self")))

        .def("setTimesteps",             &Bandit::ESRLPolicy::setTimesteps,
                 "This function sets the required number of timesteps per exploration phase.\n"
                 "\n"
                 "@param t The new number of timesteps."
        , (arg("self"), "t"))

        .def("getTimesteps",             &Bandit::ESRLPolicy::getTimesteps,
                 "This function returns the currently set number of timesteps per exploration phase.\n"
                 "\n"
                 "@return The currently set number of timesteps."
        , (arg("self")))

        .def("setExplorationPhases",     &Bandit::ESRLPolicy::setExplorationPhases,
                 "This function sets the required number of exploration phases before exploitation.\n"
                 "\n"
                 "@param p The new number of exploration phases."
        , (arg("self"), "p"))

        .def("getExplorationPhases",     &Bandit::ESRLPolicy::getExplorationPhases,
                 "This function returns the currently set number of exploration phases before exploitation.\n"
                 "\n"
                 "@return The currently set number of exploration phases."
        , (arg("self")))

        .def("setWindowSize",            &Bandit::ESRLPolicy::setWindowSize,
                 "This function sets the size of the timestep window to compute the value of the action that ESRL is converging to.\n"
                 "\n"
                 "@param window The new size of the average window."
        , (arg("self"), "p"))

        .def("getWindowSize",            &Bandit::ESRLPolicy::getWindowSize,
                 "This function returns the currently set size of the timestep window to compute the value of an action.\n"
                 "\n"
                 "@return The currently set window size."
        , (arg("self")));
}
