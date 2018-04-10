#define BOOST_TEST_MODULE FactoredMDP_SparseCooperativeQLearning
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/FactoredMDP/Algorithms/SparseCooperativeQLearning.hpp>

namespace fm = AIToolbox::FactoredMDP;

BOOST_AUTO_TEST_CASE( simple_rule_update ) {
    const fm::State S{2};
    const fm::Action A{2, 2, 2};

    const double v1 = 1.0, v3 = 3.0, v5 = 5.0, v6 = 6.0;
    std::vector<fm::QFunctionRule> rules {
        {{{0}, {0}}, {{0},    {1}},      v1},
        {{{0}, {1}}, {{0, 1}, {0, 1}},  2.0},
        {{{0}, {1}}, {{0, 1}, {1, 0}},   v3},
        {{{0}, {0}}, {{0, 1}, {1, 0}},  4.0},
        {{{0}, {0}}, {{1, 2}, {1, 1}},   v5},
        {{{0}, {1}}, {{2},    {0}},      v6}
    };

    const double alpha = 0.3, gamma = 0.9;
    fm::SparseCooperativeQLearning solver(S, A, gamma, alpha);

    for (const auto & rule : rules)
        solver.insertRule(rule);

    const auto & container = solver.getQFunctionRules().getContainer();
    BOOST_CHECK_EQUAL(container[0].value_,  v1);
    BOOST_CHECK_EQUAL(container[1].value_, 2.0);
    BOOST_CHECK_EQUAL(container[2].value_,  v3);
    BOOST_CHECK_EQUAL(container[3].value_, 4.0);
    BOOST_CHECK_EQUAL(container[4].value_,  v5);
    BOOST_CHECK_EQUAL(container[5].value_,  v6);

    const double R1 = 3.7, R2 = -1.3, R3 = 7.34;
    fm::Rewards rew(3); rew << R1, R2, R3;
    const auto a1 = solver.stepUpdateQ({0}, {1, 1, 1}, {1}, rew);

    // Verify action
    BOOST_CHECK_EQUAL(AIToolbox::veccmp(a1, fm::Action{1, 0, 0}), 0);

    // Verify rules updates (from the paper)
    BOOST_CHECK_EQUAL(container[0].value_,  v1 + alpha * (R1 + gamma * (v3 / 2.0) - v1));
    BOOST_CHECK_EQUAL(container[1].value_, 2.0);
    BOOST_CHECK_EQUAL(container[2].value_,  v3);
    BOOST_CHECK_EQUAL(container[3].value_, 4.0);
    BOOST_CHECK_EQUAL(container[4].value_, v5 + alpha * (R2 + gamma * (v3 / 2.0) - v5 / 2.0 + R3 + gamma * v6 - v5 / 2.0));
    BOOST_CHECK_EQUAL(container[5].value_,  v6);
}
