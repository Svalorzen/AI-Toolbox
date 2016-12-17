#define BOOST_TEST_MODULE FactoredMDP_SparseCooperativeQLearning
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <iostream>

#include <AIToolbox/FactoredMDP/Algorithms/SparseCooperativeQLearning.hpp>
#include <AIToolbox/FactoredMDP/Policies/EpsilonPolicy.hpp>
#include <AIToolbox/FactoredMDP/Policies/SingleActionPolicy.hpp>

#include "Utils/ToroidalWorld.hpp"

namespace fm = AIToolbox::FactoredMDP;

static std::default_random_engine testRand(AIToolbox::Impl::Seeder::getSeed());

int distance(int x1, int x2) {
    auto d = std::abs(x1 - x2);
    return std::min(d, std::abs(10 - d));
}

int mdistance(int x1, int y1, int x2, int y2) {
    return distance(x1, x2) + distance(y1, y2);
}

std::tuple<fm::State, fm::Rewards, bool> worldStep(const fm::State & s, const fm::Action & a) {
    constexpr auto captureReward = 37.5;
    constexpr auto collisionReward = -50.0;
    constexpr auto singleCaptureReward = -5.0;
    constexpr auto stepReward = -0.5;

    static ToroidalWorldState a1(10, 10, 0, 0);
    static ToroidalWorldState a2(10, 10, 0, 0);
    static bool ended = true;
    static std::uniform_int_distribution<unsigned> antelopeDist(0, 4); // Antelope actions

    if (ended) {
        a1.setX(s[0]); a1.setY(s[1]);
        a2.setX(s[2]); a2.setY(s[3]);
    }

    a1.setAdjacent((Direction)a[0]);
    a2.setAdjacent((Direction)a[1]);

    double r1 = stepReward, r2 = stepReward;

    ended = true;
    if (a1 == 0 && a2 == 0) {
        r1 = captureReward;
        r2 = captureReward;
    } else if (a1 == a2) {
        r1 = collisionReward;
        r2 = collisionReward;
    } else if (a1 == 0) {
        r1 = singleCaptureReward;
    } else if (a2 == 0) {
        r2 = singleCaptureReward;
    } else {
        // Now the antelope moves.
        auto antelope = antelopeDist(testRand);
        a1.setAdjacent((Direction)antelope);
        a2.setAdjacent((Direction)antelope);
        ended = false;
    }

    return std::make_tuple(fm::State{a1.getX(), a1.getY(), a2.getX(), a2.getY()},
                           fm::Rewards{r1, r2},
                           ended);
}

BOOST_AUTO_TEST_CASE( tigers_antelope_10x10 ) {
    constexpr auto eGreedy = 0.8; // We make random choices in 0.2 cases.
    constexpr auto alpha = 0.3;
    constexpr auto discount = 0.9;

    fm::State S{10, 10, 10, 10};
    fm::Action A{5, 5};

    fm::SparseCooperativeQLearning solver(S, A, discount, alpha);
    solver.reserveRules(495 + 495 + 31200);

    for (size_t x1 = 0; x1 < 10; ++x1)
    for (size_t y1 = 0; y1 < 10; ++y1) {
        if (x1 == 0 && y1 == 0) continue;
        for (size_t x2 = 0; x2 < 10; ++x2)
        for (size_t y2 = 0; y2 < 10; ++y2) {
            if (x1 == x2 && y1 == y2) continue;
            if (x2 == 0 && y2 == 0) continue;
            if (mdistance(x1, y1, x2, y2) <= 2 ||
                    (mdistance(x1, y1, 0, 0) <= 2 &&
                     mdistance(x2, y2, 0, 0) <= 2))
            {
                for (size_t a1 = 0; a1 < 5; ++a1)
                for (size_t a2 = 0; a2 < 5; ++a2)
                    solver.insertRule(fm::QFunctionRule{{{0, 1, 2, 3}, {x1, y1, x2, y2}}, {{0, 1}, {a1, a2}}, 75.0});
            }
        }
        for (size_t a = 0; a < 5; ++a) {
            solver.insertRule(fm::QFunctionRule{{{0, 1}, {x1, y1}}, {{0}, {a}}, 75.0});
            solver.insertRule(fm::QFunctionRule{{{2, 3}, {x1, y1}}, {{1}, {a}}, 75.0});
        }
    }

    BOOST_CHECK_EQUAL(solver.rulesSize(), 495 + 495 + 31200);

    std::vector<fm::State> starts{
        {1, 1, 9, 9},
        {4, 5, 6, 7},
        {5, 4, 5, 5},
        {2, 5, 3, 8},
        {6, 2, 3, 4},
        {8, 1, 0, 9},
        {3, 1, 1, 2},
        {7, 8, 6, 6},
        {4, 3, 7, 2},
        {8, 7, 5, 4},
        {1, 6, 2, 7},
        {8, 8, 9, 2},
        {8, 7, 1, 7},
        {5, 1, 8, 8},
        {6, 1, 7, 0},
        {3, 0, 5, 1},
        {3, 6, 9, 0},
        {1, 0, 9, 3},
        {7, 6, 9, 6},
        {5, 2, 1, 7},
    };
    std::uniform_int_distribution<size_t> startDist(0, starts.size() - 1);

    fm::SingleActionPolicy p(S, A);
    fm::EpsilonPolicy ePolicy(p, eGreedy);

    fm::State s, s1; fm::Action a; fm::Rewards rew; bool ended;
    for ( int episode = 0; episode < 10000; ++episode ) {
        s = starts[startDist(testRand)];
        for ( int i = 0; i < 10000; ++i ) {
            a = ePolicy.sampleAction(s);
            // Print s...
            std::tie(s1, rew, ended) = worldStep(s, a);

            p.updateAction(solver.stepUpdateQ(s, a, s1, rew));

            if (ended) break;
            s = std::move(s1);
        }
    }
}
