#define BOOST_TEST_MODULE Factored_Game_LRPPolicy
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Impl/Seeder.hpp>
#include <AIToolbox/Bandit/Policies/LRPPolicy.hpp>

std::pair<double, double> testPrisonersDilemma(unsigned a, unsigned b) {
    // Normalized to 1.0
    //
    // 0.5/0.5   0.0/0.9
    // 0.9/0.0   0.1/0.1

    if ( a == b && b == 0 ) return {0.5, 0.5};
    if ( a == b && b == 1 ) return {0.1, 0.1};
    if ( a == 1 ) return {0.9, 0.0};
    return {0.0, 0.9};
}

BOOST_AUTO_TEST_CASE( prisoners_dilemma ) {
    using namespace AIToolbox::Bandit;
    constexpr size_t A = 2;

    LRPPolicy p1(A, 0.05);
    LRPPolicy p2(A, 0.05);

    std::uniform_real_distribution<double> dist(0.0, 1.0);
    AIToolbox::RandomEngine rand(AIToolbox::Impl::Seeder::getSeed());

    for ( unsigned i = 0; i < 1000; ++i ) {
        const size_t a = p1.sampleAction();
        const size_t b = p2.sampleAction();

        const auto [r1, r2] = testPrisonersDilemma(a,b);

        p1.stepUpdateP(a, dist(rand) < r1);
        p2.stepUpdateP(b, dist(rand) < r2);

    }
    // Check convergence to Nash equilibrium
    BOOST_CHECK(p1.getActionProbability(1) > 0.9);
    BOOST_CHECK(p2.getActionProbability(1) > 0.9);
}

std::pair<double, double> testRandomishGame(unsigned a, unsigned b) {
    // Normalized to 1.0
    //
    // 0.5/0.5   0.5/0.5   0.5/0.5
    // 0.5/0.5   0.5/0.5   0.5/0.5
    // 0.5/0.5   0.5/0.5   0.7/0.7

    if ( a == b && b == 2 ) return {0.7, 0.7};
    return {0.5, 0.5};
}

BOOST_AUTO_TEST_CASE( randomish_game ) {
    using namespace AIToolbox::Bandit;
    constexpr size_t A = 3;

    LRPPolicy p1(A, 0.01);
    LRPPolicy p2(A, 0.01);

    std::uniform_real_distribution<double> dist(0.0, 1.0);
    AIToolbox::RandomEngine rand(AIToolbox::Impl::Seeder::getSeed());

    for ( unsigned i = 0; i < 50000; ++i ) {
        const size_t a = p1.sampleAction();
        const size_t b = p2.sampleAction();

        const auto [r1, r2] = testRandomishGame(a,b);

        p1.stepUpdateP(a, dist(rand) < r1);
        p2.stepUpdateP(b, dist(rand) < r2);
    }
    // Check convergence to Nash equilibrium
    BOOST_CHECK(p1.getActionProbability(2) > 0.9);
    BOOST_CHECK(p2.getActionProbability(2) > 0.9);
}
