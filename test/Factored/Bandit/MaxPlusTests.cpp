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
    auto [A, workers, minePs] = fb::makeMiningParameters(0);

    std::cout << "There are " << A.size() << " villages.\n";
    std::cout << " - " << A << '\n';
    std::cout << "The villages have the following worker counts:\n";
    std::cout << " - " << workers << '\n';
    std::cout << "There are " << minePs.size() << " mines.\n";
    std::cout << "The mines have the following productivities:\n";
    std::cout << " - " << minePs << '\n';

    fb::MiningBandit bandit(A, workers, minePs);

    std::cout << "Each mine is connected to the following villages:\n";
    for (const auto & m : bandit.getGroups())
        std::cout << " - " << m << '\n';

    std::cout << "The optimal action for the problem is: " << bandit.getOptimalAction() << '\n';

    const auto & groups = bandit.getGroups();
    std::vector<fb::QFunctionRule> rules;
    for (size_t m = 0; m < groups.size(); ++m) {
        const auto & mineVillages = groups[m];

        aif::PartialFactorsEnumerator enumerator(A, mineVillages);
        while (enumerator.isValid()) {
            const auto & villageAction = enumerator->second;

            unsigned totalMiners = 0;
            for (size_t v = 0; v < mineVillages.size(); ++v)
                if (mineVillages[v] + villageAction[v] == m)
                    totalMiners += workers[mineVillages[v]];

            if (totalMiners > 0) {
                const double v = rewFun(minePs[m], totalMiners);
                rules.emplace_back(*enumerator, v);
            }

            enumerator.advance();
        }
    }

    MP mp;
    const auto [bestAction, val] = mp(A, rules);

    std::cout << "MaxPlus thinks that the optimal action for the problem is: " << bestAction << '\n';
    std::cout << "- Regret for this action is: " << bandit.getRegret(bestAction) << '\n';

    double minRegret = std::numeric_limits<double>::max();
    double maxRegret = std::numeric_limits<double>::lowest();
    double avgRegret = 0.0;

    AIToolbox::RandomEngine eng(100);
    constexpr size_t toTry = 100;
    for (size_t i = 0; i < toTry; ++i) {
        const auto action = aif::makeRandomValue(A, eng);

        const auto regret = bandit.getRegret(action);

        minRegret = std::min(minRegret, regret);
        maxRegret = std::max(maxRegret, regret);
        avgRegret += regret;
    }
    avgRegret /= toTry;

    std::cout << "I also tried " << toTry << " random actions to see how it compares.\n";
    std::cout << " - Min regret: " << minRegret << '\n';
    std::cout << " - Max regret: " << maxRegret << '\n';
    std::cout << " - Avg regret: " << avgRegret << '\n';
}
