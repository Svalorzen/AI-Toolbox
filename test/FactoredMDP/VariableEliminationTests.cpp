#define BOOST_TEST_MODULE FactoredMDP_VariableElimination
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <iostream>

#include <AIToolbox/FactoredMDP/Algorithms/VariableElimination.hpp>

namespace fm = AIToolbox::FactoredMDP;
using ve = fm::VariableElimination;

BOOST_AUTO_TEST_CASE( simple_graph ) {
    std::vector<fm::QFunctionRule> rules {
        // States, Actions,                     Value
        {    {},   {{0, 2}, {1, 0}},            4.0},
        {    {},   {{0, 1}, {1, 0}},            5.0},
        {    {},   {{1},    {0}},               2.0},
        {    {},   {{1, 2}, {1, 1}},            5.0},
    };

    auto solution = std::make_pair(fm::Action{1, 0, 0}, 11.0);

    fm::Action a{2, 2, 2};

    ve v(a);
    auto bestAction_v = v(rules);

    BOOST_CHECK_EQUAL(bestAction_v.second, solution.second);
    BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(bestAction_v.first), std::end(bestAction_v.first),
                                  std::begin(solution.first), std::end(solution.first));
}
