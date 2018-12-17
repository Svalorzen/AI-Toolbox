#define BOOST_TEST_MODULE Bandit_ThompsonSamplingPolicy
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <array>
#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Bandit/Algorithms/RollingAverage.hpp>
#include <AIToolbox/Bandit/Policies/ThompsonSamplingPolicy.hpp>

BOOST_AUTO_TEST_CASE( sampling ) {
    using namespace AIToolbox;
    constexpr size_t A = 3;

    Bandit::RollingAverage ra(A);
    Bandit::ThompsonSamplingPolicy p(ra.getQFunction(), ra.getCounts());

    std::array<unsigned, A> counts{{0,0,0}};
    for (unsigned i = 0; i < 1000; ++i)
        ++counts[p.sampleAction()];

    // We shouldn't have picked 1, and 0/2 should be approximately equal
    BOOST_CHECK(counts[0] > 200);
    BOOST_CHECK(counts[1] > 200);
    BOOST_CHECK(counts[2] > 200);

    ra.stepUpdateQ(1, 1.0);
    ra.stepUpdateQ(1, 1.0);
    ra.stepUpdateQ(1, 1.0);
    ra.stepUpdateQ(1, 1.0);
    ra.stepUpdateQ(1, 1.0);
    ra.stepUpdateQ(2, 1.0);
    ra.stepUpdateQ(2, 1.0);
    ra.stepUpdateQ(2, 1.0);
    ra.stepUpdateQ(2, 1.0);
    ra.stepUpdateQ(2, 1.0);

    // Reset counts
    std::fill(std::begin(counts), std::end(counts), 0);

    for (unsigned i = 0; i < 1000; ++i)
        ++counts[p.sampleAction()];

    BOOST_CHECK(100 < counts[0] && counts[0] < 180);
    BOOST_CHECK(375 < counts[1] && counts[1] < 485);
    BOOST_CHECK(375 < counts[2] && counts[2] < 485);
}

BOOST_AUTO_TEST_CASE( probability ) {
    using namespace AIToolbox;
    constexpr size_t A = 3;

    Bandit::RollingAverage ra(A);
    Bandit::ThompsonSamplingPolicy p(ra.getQFunction(), ra.getCounts());

    for (unsigned i = 0; i < A; ++i) {
        const auto pp = p.getActionProbability(i);
        BOOST_CHECK(0.30 < pp && pp < 0.35);
    }

    ra.stepUpdateQ(1, 1.0);
    ra.stepUpdateQ(1, 1.0);
    ra.stepUpdateQ(1, 1.0);
    ra.stepUpdateQ(1, 1.0);
    ra.stepUpdateQ(1, 1.0);
    ra.stepUpdateQ(2, 1.0);
    ra.stepUpdateQ(2, 1.0);
    ra.stepUpdateQ(2, 1.0);
    ra.stepUpdateQ(2, 1.0);
    ra.stepUpdateQ(2, 1.0);

    const auto p0 = p.getActionProbability(0);
    const auto p1 = p.getActionProbability(1);
    const auto p2 = p.getActionProbability(2);
    const auto pol = p.getPolicy();

    BOOST_CHECK(0.100 < p0 && p0 < 0.180);
    BOOST_CHECK(0.375 < p1 && p1 < 0.485);
    BOOST_CHECK(0.375 < p2 && p2 < 0.485);

    BOOST_CHECK(0.100 < pol[0] && pol[0] < 0.180);
    BOOST_CHECK(0.375 < pol[1] && pol[1] < 0.485);
    BOOST_CHECK(0.375 < pol[2] && pol[2] < 0.485);
}
