#define BOOST_TEST_MODULE FactoredMDP_LLR

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Impl/Seeder.hpp>
#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/FactoredMDP/Utils.hpp>
#include <AIToolbox/FactoredMDP/Algorithms/LLR.hpp>
#include <AIToolbox/FactoredMDP/Policies/QGreedyPolicy.hpp>

namespace fm = AIToolbox::FactoredMDP;

BOOST_AUTO_TEST_CASE( xxx_simple_example_small ) {
    fm::Action A{2,2,2};
    fm::LLR llr(A, {{0,1}, {1,2}});

    // Two rewards since we have two agent groups.
    fm::Rewards rew(2);

    auto getEvenReward = [](size_t a1, size_t a2){
        static std::default_random_engine rand(0);
        constexpr double factorsNum = 2.0;
        if (!a1 && !a2) {
            std::bernoulli_distribution roll(0.75);
            return roll(rand) / factorsNum;
        } else if (!a1 && a2) {
            return 1.0 / factorsNum;
        } else if (a1 && !a2) {
            std::bernoulli_distribution roll(0.25);
            return roll(rand) / factorsNum;
        } else {
            std::bernoulli_distribution roll(0.9);
            return roll(rand) / factorsNum;
        }
    };

    auto getOddReward = [](size_t a1, size_t a2){
        static std::default_random_engine rand(1);
        constexpr double factorsNum = 2.0;
        if (!a1 && !a2) {
            std::bernoulli_distribution roll(0.75);
            return roll(rand) / factorsNum;
        } else if (!a1 && a2) {
            std::bernoulli_distribution roll(0.25);
            return roll(rand) / factorsNum;
        } else if (a1 && !a2) {
            return 1.0 / factorsNum;
        } else {
            std::bernoulli_distribution roll(0.9);
            return roll(rand) / factorsNum;
        }
    };

    fm::Action action{0,0,0};
    for (unsigned t = 0; t < 10000; ++t) {
        rew[0] = getEvenReward(action[0], action[1]);
        rew[1] = getOddReward(action[1], action[2]);

        action = llr.stepUpdateQ(action, rew);
    }

    fm::Action solution{0, 1, 0};

    auto rules = llr.getQFunctionRules();
    fm::QGreedyPolicy p({}, A, rules);

    auto greedyAction = p.sampleAction({});

    BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(solution), std::end(solution),
                                  std::begin(greedyAction), std::end(greedyAction));
}
