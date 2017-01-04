#define BOOST_TEST_MODULE FactoredMDP_VariableElimination
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/FactoredMDP/Algorithms/Utils/VariableElimination.hpp>

namespace fm = AIToolbox::FactoredMDP;
using VE = fm::VariableElimination;

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

    VE v(a);
    auto bestAction_v = v(rules);

    BOOST_CHECK_EQUAL(std::get<1>(bestAction_v), std::get<1>(solution));
    BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(std::get<0>(bestAction_v)), std::end(std::get<0>(bestAction_v)),
                                  std::begin(std::get<0>(solution)),     std::end(std::get<0>(solution)));
}

BOOST_AUTO_TEST_CASE( all_unconnected_agents ) {
    std::vector<fm::QFunctionRule> rules {
        // States, Actions,                     Value
        {    {},   {{0},    {2}},               4.0},
        {    {},   {{1},    {0}},               2.0},
        {    {},   {{2},    {0}},               3.0},
        {    {},   {{3},    {1}},               7.0},
    };

    auto solution = std::make_pair(fm::Action{2, 0, 0, 1}, 16.0);

    fm::Action a{3, 2, 3, 4};

    VE v(a);
    auto bestAction_v = v(rules);

    BOOST_CHECK_EQUAL(std::get<1>(bestAction_v), std::get<1>(solution));
    BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(std::get<0>(bestAction_v)), std::end(std::get<0>(bestAction_v)),
                                  std::begin(std::get<0>(solution)),     std::end(std::get<0>(solution)));
}

BOOST_AUTO_TEST_CASE( all_connected_agents ) {
    std::vector<fm::QFunctionRule> rules {
        // States, Actions,                     Value
        {    {},   {{0, 1, 2}, {1, 1, 1}},      10.0},
    };

    auto solution = std::make_pair(fm::Action{1, 1, 1}, 10.0);

    fm::Action a{2, 2, 2};

    VE v(a);
    auto bestAction_v = v(rules);

    BOOST_CHECK_EQUAL(std::get<1>(bestAction_v), std::get<1>(solution));
    BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(std::get<0>(bestAction_v)), std::end(std::get<0>(bestAction_v)),
                                  std::begin(std::get<0>(solution)),     std::end(std::get<0>(solution)));
}

BOOST_AUTO_TEST_CASE( negative_graph ) {
    std::vector<fm::QFunctionRule> rules {
        // States, Actions,                     Value
        {    {},   {{0}, {0}},                 -10.0},
        // We must explicitly mention this rule since the this agent has at
        // least one negative rule
        {    {},   {{0}, {1}},                   0.0},
        // Here we don't have to mention them all, since the negative rule only
        // concerned agent 0
        {    {},   {{0, 1}, {0, 0}},            11.0},
    };

    auto solution = std::make_pair(fm::Action{0, 0}, 1.0);

    fm::Action a{2, 2};

    VE v(a);
    auto bestAction_v = v(rules);

    BOOST_CHECK_EQUAL(std::get<1>(bestAction_v), std::get<1>(solution));
    BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(std::get<0>(bestAction_v)), std::end(std::get<0>(bestAction_v)),
                                  std::begin(std::get<0>(solution)),     std::end(std::get<0>(solution)));
}
