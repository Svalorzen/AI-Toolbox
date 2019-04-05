#define BOOST_TEST_MODULE Factored_Bandit_VariableElimination
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Factored/Bandit/Algorithms/Utils/VariableElimination.hpp>

namespace aif = AIToolbox::Factored;
namespace fb = AIToolbox::Factored::Bandit;
using VE = fb::VariableElimination;

BOOST_AUTO_TEST_CASE( simple_graph ) {
    const std::vector<fb::QFunctionRule> rules {
        // Actions,                     Value
        {  {{0, 2}, {1, 0}},            4.0},
        {  {{0, 1}, {1, 0}},            5.0},
        {  {{1},    {0}},               2.0},
        {  {{1, 2}, {1, 1}},            5.0},
    };

    const auto solA = aif::Action{1, 0, 0};
    const auto solV = 11.0;

    const aif::Action a{2, 2, 2};

    VE v;
    const auto [bestAction, val] = v(a, rules);

    BOOST_CHECK_EQUAL(val, solV);
    BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(bestAction), std::end(bestAction),
                                  std::begin(solA),     std::end(solA));
}

BOOST_AUTO_TEST_CASE( all_unconnected_agents ) {
    const std::vector<fb::QFunctionRule> rules {
        // Actions,                     Value
        {  {{0},    {2}},               4.0},
        {  {{1},    {0}},               2.0},
        {  {{2},    {0}},               3.0},
        {  {{3},    {1}},               7.0},
    };

    const auto solA = aif::Action{2, 0, 0, 1};
    const auto solV = 16.0;

    const aif::Action a{3, 2, 3, 4};

    VE v;
    const auto [bestAction, val] = v(a, rules);

    BOOST_CHECK_EQUAL(val, solV);
    BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(bestAction), std::end(bestAction),
                                  std::begin(solA),     std::end(solA));
}

BOOST_AUTO_TEST_CASE( all_connected_agents ) {
    const std::vector<fb::QFunctionRule> rules {
        // Actions,                     Value
        {  {{0, 1, 2}, {1, 1, 1}},      10.0},
    };

    const auto solA = aif::Action{1, 1, 1};
    const auto solV = 10.0;

    const aif::Action a{2, 2, 2};

    VE v;
    const auto [bestAction, val] = v(a, rules);

    BOOST_CHECK_EQUAL(val, solV);
    BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(bestAction), std::end(bestAction),
                                  std::begin(solA),     std::end(solA));
}

BOOST_AUTO_TEST_CASE( negative_graph_1 ) {
    const std::vector<fb::QFunctionRule> rules {
        // Actions,                     Value
        {  {{0}, {0}},                 -10.0},
        // We must explicitly mention this rule since the this agent has at
        // least one negative rule
        {  {{0}, {1}},                   0.0},
        // Here we don't have to mention them all, since the negative rule only
        // concerned agent 0
        {  {{0, 1}, {0, 0}},            11.0},
    };

    const auto solA = aif::Action{0, 0};
    const auto solV = 1.0;

    const aif::Action a{2, 2};

    VE v;
    const auto [bestAction, val] = v(a, rules);

    BOOST_CHECK_EQUAL(val, solV);
    BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(bestAction), std::end(bestAction),
                                  std::begin(solA),     std::end(solA));
}

BOOST_AUTO_TEST_CASE( negative_graph_2 ) {
    const std::vector<fb::QFunctionRule> rules {
        // Actions,                     Value
        {  {{0}, {0}},                 -10.0},
        // We must explicitly mention this rule since the this agent has at
        // least one negative rule
        {  {{0}, {1}},                   0.0},
        // Here we don't have to mention them all, since the negative rule only
        // concerned agent 0
        {  {{0, 1}, {0, 0}},             9.0},
    };

    const auto solA = aif::Action{1, 0};
    const auto solV = 0.0;

    const aif::Action a{2, 2};

    VE v;
    const auto [bestAction, val] = v(a, rules);

    BOOST_CHECK_EQUAL(val, solV);
    BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(bestAction), std::end(bestAction),
                                  std::begin(solA),     std::end(solA));
}
