#define BOOST_TEST_MODULE Factored_Bandit_LLR

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Impl/Seeder.hpp>
#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>
#include <AIToolbox/Factored/Bandit/Algorithms/LLR.hpp>
#include <AIToolbox/Factored/Bandit/Policies/QGreedyPolicy.hpp>

namespace aif = AIToolbox::Factored;
namespace fb = AIToolbox::Factored::Bandit;

BOOST_AUTO_TEST_CASE( xxx_simple_example_small ) {
    aif::Action A{2,2,2};
    fb::LLR llr(A, {{0,1}, {1,2}});

    // Two rewards since we have two agent groups.
    aif::Rewards rew(2);

    auto getEvenReward = [](size_t a1, size_t a2){
        static AIToolbox::RandomEngine rand(0);
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
        static AIToolbox::RandomEngine rand(1);
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

    aif::Action action{0,0,0};
    for (unsigned t = 0; t < 10000; ++t) {
        rew[0] = getEvenReward(action[0], action[1]);
        rew[1] = getOddReward(action[1], action[2]);

        action = llr.stepUpdateQ(action, rew);
    }

    const aif::Action solution{0, 1, 0};

    const auto q = llr.getRollingAverage().getQFunction();
    fb::QGreedyPolicy p(A, q);

    const auto greedyAction = p.sampleAction();

    BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(solution), std::end(solution),
                                  std::begin(greedyAction), std::end(greedyAction));
}
