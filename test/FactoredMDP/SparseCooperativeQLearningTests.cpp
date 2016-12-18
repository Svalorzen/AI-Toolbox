#define BOOST_TEST_MODULE FactoredMDP_SparseCooperativeQLearning
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/FactoredMDP/Algorithms/SparseCooperativeQLearning.hpp>

namespace fm = AIToolbox::FactoredMDP;

BOOST_AUTO_TEST_CASE( simple_rule_update ) {
    fm::State S{2};
    fm::Action A{2, 2, 2};

    double v1 = 1.0, v3 = 3.0, v5 = 5.0, v6 = 6.0;
    std::vector<fm::QFunctionRule> rules {
        {{{0}, {0}}, {{0},    {1}},      v1},
        {{{0}, {1}}, {{0, 1}, {0, 1}},  2.0},
        {{{0}, {1}}, {{0, 1}, {1, 0}},   v3},
        {{{0}, {0}}, {{0, 1}, {1, 0}},  4.0},
        {{{0}, {0}}, {{1, 2}, {1, 1}},   v5},
        {{{0}, {1}}, {{2},    {0}},      v6}
    };

    double alpha = 0.3, gamma = 0.9;
    fm::SparseCooperativeQLearning solver(S, A, gamma, alpha);

    for (const auto & rule : rules)
        solver.insertRule(rule);

    double R1 = 3.7, R2 = -1.3, R3 = 7.34;
    solver.stepUpdateQ({0}, {1, 1, 1}, {1}, {R1, R2, R3});

    const auto & container = solver.getQFunctionRules();

    auto r1Iterable = container.filter({0, 1, 1, 0});
    BOOST_CHECK_EQUAL(r1Iterable.size(), 1);

    auto r1 = *r1Iterable.begin();
    // This is taken from the paper
    BOOST_CHECK_EQUAL(r1.value_, v1 + alpha * (R1 + gamma * (v3 / 2.0) - v1));

    auto r5Iterable = container.filter({0, 0, 1, 1});
    BOOST_CHECK_EQUAL(r5Iterable.size(), 1);

    auto r5 = *r5Iterable.begin();
    // This is taken from the paper
    BOOST_CHECK_EQUAL(r5.value_, v5 + alpha * (R2 + gamma * (v3 / 2.0) - v5 / 2.0 + R3 + gamma * v6 - v5 / 2.0));
}
