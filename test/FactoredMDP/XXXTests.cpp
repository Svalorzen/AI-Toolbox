#define BOOST_TEST_MODULE FactoredMDP_XXX

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Impl/Seeder.hpp>
#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/FactoredMDP/Utils.hpp>
#include <AIToolbox/FactoredMDP/Algorithms/XXXAlgorithm.hpp>

namespace fm = AIToolbox::FactoredMDP;

BOOST_AUTO_TEST_CASE( xxx_simple_example_small ) {
    fm::Action a{2,2,2};
    fm::XXXAlgorithm x(a, {{1.0, {0,1}}, {1.0, {1,2}}});

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
        printAction(action);
        rew[0] = getEvenReward(action[0], action[1]);
        rew[1] = getOddReward(action[1], action[2]);
        std::cout << " ==> " << rew[0] << ", " << rew[1] << "\n";

        action = x.stepUpdateQ(action, rew);
    }
}
