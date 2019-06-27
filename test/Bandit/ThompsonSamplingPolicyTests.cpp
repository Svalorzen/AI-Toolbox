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
    Bandit::ThompsonSamplingPolicy p(ra.getQFunction(), ra.getM2s(), ra.getCounts());

    std::array<unsigned, A> counts{{0,0,0}};

    // We must give at least a couple of observation per arm, and with some
    // spread
    ra.stepUpdateQ(0, -0.5);
    ra.stepUpdateQ(0, 0.5);
    ra.stepUpdateQ(1, 1.5);
    ra.stepUpdateQ(1, 2.0);
    ra.stepUpdateQ(1, 0.5);
    ra.stepUpdateQ(1, 0.0);
    ra.stepUpdateQ(1, 1.0);
    ra.stepUpdateQ(2, 1.5);
    ra.stepUpdateQ(2, 2.0);
    ra.stepUpdateQ(2, 0.5);
    ra.stepUpdateQ(2, 0.0);
    ra.stepUpdateQ(2, 1.0);

    for (unsigned i = 0; i < 1000; ++i)
        ++counts[p.sampleAction()];

    BOOST_TEST_INFO("counts[0] = " << counts[0]);
    BOOST_CHECK(50 < counts[0] && counts[0] < 180);
    BOOST_TEST_INFO("counts[1] = " << counts[1]);
    BOOST_CHECK(375 < counts[1] && counts[1] < 485);
    BOOST_TEST_INFO("counts[2] = " << counts[2]);
    BOOST_CHECK(375 < counts[2] && counts[2] < 485);
}

BOOST_AUTO_TEST_CASE( probability ) {
    using namespace AIToolbox;
    constexpr size_t A = 3;

    Bandit::RollingAverage ra(A);
    Bandit::ThompsonSamplingPolicy p(ra.getQFunction(), ra.getM2s(), ra.getCounts());

    // We must give at least a couple of observation per arm, and with some
    // spread
    ra.stepUpdateQ(0, -0.5);
    ra.stepUpdateQ(0, 0.5);
    ra.stepUpdateQ(1, 1.5);
    ra.stepUpdateQ(1, 2.0);
    ra.stepUpdateQ(1, 0.5);
    ra.stepUpdateQ(1, 0.0);
    ra.stepUpdateQ(1, 1.0);
    ra.stepUpdateQ(2, 1.5);
    ra.stepUpdateQ(2, 2.0);
    ra.stepUpdateQ(2, 0.5);
    ra.stepUpdateQ(2, 0.0);
    ra.stepUpdateQ(2, 1.0);

    const auto p0 = p.getActionProbability(0);
    const auto p1 = p.getActionProbability(1);
    const auto p2 = p.getActionProbability(2);
    const auto pol = p.getPolicy();

    BOOST_CHECK(0.050 < p0 && p0 < 0.180);
    BOOST_CHECK(0.375 < p1 && p1 < 0.485);
    BOOST_CHECK(0.375 < p2 && p2 < 0.485);

    BOOST_CHECK(0.050 < pol[0] && pol[0] < 0.180);
    BOOST_CHECK(0.375 < pol[1] && pol[1] < 0.485);
    BOOST_CHECK(0.375 < pol[2] && pol[2] < 0.485);
}
