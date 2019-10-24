#define BOOST_TEST_MODULE Bandit_QGreedyPolicy
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <array>
#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Bandit/Experience.hpp>
#include <AIToolbox/Bandit/Policies/QGreedyPolicy.hpp>

BOOST_AUTO_TEST_CASE( sampling ) {
    using namespace AIToolbox;
    constexpr size_t A = 3;

    Bandit::Experience exp(A);
    Bandit::QGreedyPolicy p(exp.getRewardMatrix());

    std::array<unsigned, A> counts{{0,0,0}};
    for (unsigned i = 0; i < 1000; ++i)
        ++counts[p.sampleAction()];

    // We shouldn't have picked 1, and 0/2 should be approximately equal
    BOOST_CHECK(counts[0] > 200);
    BOOST_CHECK(counts[1] > 200);
    BOOST_CHECK(counts[2] > 200);

    exp.record(1, 1.0);
    exp.record(2, 1.0);

    // Reset counts
    std::fill(std::begin(counts), std::end(counts), 0);

    for (unsigned i = 0; i < 1000; ++i)
        ++counts[p.sampleAction()];

    BOOST_CHECK_EQUAL(counts[0], 0);
    BOOST_CHECK(counts[1] > 350);
    BOOST_CHECK(counts[2] > 350);
}

BOOST_AUTO_TEST_CASE( probability ) {
    using namespace AIToolbox;
    constexpr size_t A = 3;

    Bandit::Experience exp(A);
    Bandit::QGreedyPolicy p(exp.getRewardMatrix());

    for (unsigned i = 0; i < A; ++i)
        BOOST_CHECK_EQUAL(p.getActionProbability(i), 1.0 / A);

    exp.record(1, 1.0);
    exp.record(2, 1.0);

    BOOST_CHECK_EQUAL(p.getActionProbability(0), 0.0);
    BOOST_CHECK_EQUAL(p.getActionProbability(1), 0.5);
    BOOST_CHECK_EQUAL(p.getActionProbability(2), 0.5);
}
