#define BOOST_TEST_MODULE Factored_Bandit_MaxPlus
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include "GlobalFixtures.hpp"

#include <AIToolbox/Factored/Bandit/Algorithms/Utils/MaxPlus.hpp>
#include <AIToolbox/Factored/Bandit/Environments/MiningProblem.hpp>

#include <iostream>

namespace aif = AIToolbox::Factored;
namespace fb = AIToolbox::Factored::Bandit;
using MP = fb::MaxPlus;

// Note that in these tests we don't check the value (as we do in
// VariableElimination) because MaxPlus is an approximate algorithm (at least
// since our implementation is for loopy graphs and not trees). This makes the
// outputted values not necessarily correct, so there's little point in testing
// them. As long as the actions are correct, we should be fine.

BOOST_AUTO_TEST_CASE( simple_graph ) {
    const std::vector<fb::QFunctionRule> rules {
        // Actions,                     Value
        {  {{0, 2}, {1, 0}},            4.0},
        {  {{0, 1}, {1, 0}},            5.0},
        {  {{1},    {0}},               2.0},
        {  {{1, 2}, {1, 1}},            5.0},
    };

    const auto solA = aif::Action{1, 0, 0};
    // const auto solV = 11.0;

    const aif::Action a{2, 2, 2};

    MP mp;
    const auto [bestAction, val] = mp(a, rules);
    (void)val;

    // BOOST_CHECK_EQUAL(val, solV);
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
    // const auto solV = 16.0;

    const aif::Action a{3, 2, 3, 4};

    MP mp;
    const auto [bestAction, val] = mp(a, rules);
    (void)val;

    // BOOST_CHECK_EQUAL(val, solV);
    BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(bestAction), std::end(bestAction),
                                  std::begin(solA),     std::end(solA));
}

BOOST_AUTO_TEST_CASE( all_connected_agents ) {
    const std::vector<fb::QFunctionRule> rules {
        // Actions,                     Value
        {  {{0, 1, 2}, {1, 1, 1}},      10.0},
    };

    const auto solA = aif::Action{1, 1, 1};
    // const auto solV = 10.0;

    const aif::Action a{2, 2, 2};

    MP mp;
    const auto [bestAction, val] = mp(a, rules);
    (void)val;

    // BOOST_CHECK_EQUAL(val, solV);
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
    // const auto solV = 1.0;

    const aif::Action a{2, 2};

    MP mp;
    const auto [bestAction, val] = mp(a, rules);
    (void)val;

    // BOOST_CHECK_EQUAL(val, solV);
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
    // const auto solV = 0.0;

    const aif::Action a{2, 2};

    MP mp;
    const auto [bestAction, val] = mp(a, rules);
    (void)val;

    // BOOST_CHECK_EQUAL(val, solV);
    BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(bestAction), std::end(bestAction),
                                  std::begin(solA),     std::end(solA));
}

template <typename T>
std::ostream & operator<<(std::ostream & os, const std::vector<T> & v) {
    std::cout << '[';
    for (size_t i = 0; i < v.size() - 1; ++i)
        std::cout << v[i] << ", ";
    std::cout << v.back() << ']';
    return os;
}

double rewFun(double productivity, size_t totalWorkers) {
    assert(totalWorkers > 0);
    return productivity * std::pow(1.03, totalWorkers);
};

BOOST_AUTO_TEST_CASE( mining_problem ) {
    auto [A, workers, minePs] = fb::makeMiningParameters(10);

    fb::MiningBandit bandit(A, workers, minePs);
    const auto solA = bandit.getOptimalAction();

    const auto rules = bandit.getDeterministicRules();

    MP mp;
    const auto [bestAction, val] = mp(A, rules);

    // Note that MaxPlus is not guaranteed to find the best action!
    // In this case it does, but with other problem seeds it does not.
    //
    // In any case, we check this one and that's all we can really do.
    BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(bestAction), std::end(bestAction),
                                  std::begin(solA),     std::end(solA));
}
