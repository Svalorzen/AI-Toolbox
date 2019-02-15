#define BOOST_TEST_MODULE MDP_QGreedyPolicy
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <array>
#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/MDP/Utils.hpp>
#include <AIToolbox/MDP/Policies/QGreedyPolicy.hpp>

BOOST_AUTO_TEST_CASE( sampling ) {
    using namespace AIToolbox;
    using namespace AIToolbox::MDP;
    constexpr size_t S = 3, A = 3;

    auto q = makeQFunction(S, A);
    q(0,0) = 45;
    q(0,1) = 14;
    q(0,2) = -15;

    q(1,0) = 1001;
    q(1,1) = 1000.99;
    q(1,2) = 1001;

    q(2,0) = 42;
    q(2,1) = 42;
    q(2,2) = 42;

    QGreedyPolicy p(q);

    for (unsigned i = 0; i < 1000; ++i)
        BOOST_CHECK_EQUAL(p.sampleAction(0), 0);

    std::array<unsigned, A> counts{{0,0,0}};
    for (unsigned i = 0; i < 1000; ++i)
        ++counts[p.sampleAction(1)];

    // We shouldn't have picked 1, and 0/2 should be approximately equal
    BOOST_CHECK_EQUAL(counts[1], 0);
    BOOST_CHECK(counts[0] > 350);
    BOOST_CHECK(counts[2] > 350);

    // Reset counts
    std::fill(std::begin(counts), std::end(counts), 0);

    for (unsigned i = 0; i < 1000; ++i)
        ++counts[p.sampleAction(2)];

    // Should have picked all three.
    BOOST_CHECK(counts[0] > 200);
    BOOST_CHECK(counts[1] > 200);
    BOOST_CHECK(counts[2] > 200);
}

BOOST_AUTO_TEST_CASE( getActionProbability ) {
    using namespace AIToolbox;
    using namespace AIToolbox::MDP;
    constexpr size_t S = 3, A = 3;

    auto q = makeQFunction(S, A);
    q(0,0) = 45;
    q(0,1) = 14;
    q(0,2) = -15;

    q(1,0) = 1001;
    q(1,1) = 1000.99;
    q(1,2) = 1001;

    q(2,0) = 42;
    q(2,1) = 42;
    q(2,2) = 42;

    QGreedyPolicy p(q);

    BOOST_CHECK(checkEqualSmall(p.getActionProbability(0,0), 1.0));
    BOOST_CHECK(checkEqualSmall(p.getActionProbability(0,1), 0.0));
    BOOST_CHECK(checkEqualSmall(p.getActionProbability(0,2), 0.0));

    BOOST_CHECK(checkEqualSmall(p.getActionProbability(1,0), 0.5));
    BOOST_CHECK(checkEqualSmall(p.getActionProbability(1,1), 0.0));
    BOOST_CHECK(checkEqualSmall(p.getActionProbability(1,2), 0.5));

    BOOST_CHECK(checkEqualSmall(p.getActionProbability(2,0), 1.0/3.0));
    BOOST_CHECK(checkEqualSmall(p.getActionProbability(2,1), 1.0/3.0));
    BOOST_CHECK(checkEqualSmall(p.getActionProbability(2,2), 1.0/3.0));
}

BOOST_AUTO_TEST_CASE( getPolicy ) {
    using namespace AIToolbox;
    using namespace AIToolbox::MDP;
    constexpr size_t S = 3, A = 3;

    auto q = makeQFunction(S, A);
    q(0,0) = 45;
    q(0,1) = 14;
    q(0,2) = -15;

    q(1,0) = 1001;
    q(1,1) = 1000.99;
    q(1,2) = 1001;

    q(2,0) = 42;
    q(2,1) = 42;
    q(2,2) = 42;

    QGreedyPolicy p(q);
    const auto matrix = p.getPolicy();

    BOOST_CHECK(checkEqualSmall(matrix(0,0), 1.0));
    BOOST_CHECK(checkEqualSmall(matrix(0,1), 0.0));
    BOOST_CHECK(checkEqualSmall(matrix(0,2), 0.0));

    BOOST_CHECK(checkEqualSmall(matrix(1,0), 0.5));
    BOOST_CHECK(checkEqualSmall(matrix(1,1), 0.0));
    BOOST_CHECK(checkEqualSmall(matrix(1,2), 0.5));

    BOOST_CHECK(checkEqualSmall(matrix(2,0), 1.0/3.0));
    BOOST_CHECK(checkEqualSmall(matrix(2,1), 1.0/3.0));
    BOOST_CHECK(checkEqualSmall(matrix(2,2), 1.0/3.0));
}
