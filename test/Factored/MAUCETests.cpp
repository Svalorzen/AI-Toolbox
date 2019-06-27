#define BOOST_TEST_MODULE Factored_Bandit_MAUCE

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Impl/Seeder.hpp>
#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Factored/Bandit/Algorithms/MAUCE.hpp>
#include <AIToolbox/Factored/Bandit/Policies/QGreedyPolicy.hpp>

namespace fm = AIToolbox::Factored;
namespace fb = fm::Bandit;

BOOST_AUTO_TEST_CASE( simple_example_small ) {
    fm::Action A{2,2,2};
    fb::MAUCE x(A, {{0,1}, {1,2}}, {1.0, 1.0});

    // Two rewards since we have two agent groups.
    fm::Rewards rew(2);

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

    fm::Action action{0,0,0};
    for (unsigned t = 0; t < 10000; ++t) {
        rew[0] = getEvenReward(action[0], action[1]);
        rew[1] = getOddReward(action[1], action[2]);

        action = x.stepUpdateQ(action, rew);
    }

    const fm::Action solution{0, 1, 0};

    const auto q = x.getRollingAverage().getQFunction();
    fb::QGreedyPolicy p(A, q);

    const auto greedyAction = p.sampleAction();

    BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(solution), std::end(solution),
                                  std::begin(greedyAction), std::end(greedyAction));
}
