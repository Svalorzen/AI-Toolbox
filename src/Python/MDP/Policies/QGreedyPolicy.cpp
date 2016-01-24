#include <AIToolbox/MDP/Policies/QGreedyPolicy.hpp>

#include <boost/python.hpp>

void exportQGreedyPolicy() {
    using namespace AIToolbox::MDP;
    using namespace boost::python;

    class_<QGreedyPolicy, bases<QPolicyInterface>>{"QGreedyPolicy", init<const QFunction &>()};
}

