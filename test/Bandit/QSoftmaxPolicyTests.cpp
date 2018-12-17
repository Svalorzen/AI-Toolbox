#define BOOST_TEST_MODULE Bandit_QSoftmaxPolicy
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Utils/Core.hpp>

#include <AIToolbox/Bandit/Algorithms/RollingAverage.hpp>
#include <AIToolbox/Bandit/Policies/QSoftmaxPolicy.hpp>

BOOST_AUTO_TEST_CASE( sampling ) {
    using namespace AIToolbox;
    constexpr size_t A = 3;
    constexpr double d = 2.0;

    Bandit::RollingAverage ra(A);
    Bandit::QSoftmaxPolicy p(ra.getQFunction(), d);

    ra.stepUpdateQ(0, 10);
    ra.stepUpdateQ(1, 11);
    ra.stepUpdateQ(2, 12);

    double p0 = std::exp(10 / d);
    double p1 = std::exp(11 / d);
    double p2 = std::exp(12 / d);
    double sum = p0 + p1 + p2;

    p0 /= sum;
    p1 /= sum;
    p2 /= sum;

    std::vector<double> p3{p0, p1, p2};
    for (size_t a = 0; a < A; ++a) {
        const double ap = p.getActionProbability(a);
        BOOST_TEST_INFO("a: " << a << "; getActionProbability:" << ap << "; Solution: " << p3[a]);
        BOOST_CHECK(checkEqualSmall(ap, p3[a]));
    }

    auto pp = p.getPolicy();

    for (size_t a = 0; a < A; ++a) {
        BOOST_TEST_INFO("a: " << a << "; Policy:" << pp[a] << "; Solution: " << p3[a]);
        BOOST_CHECK(checkEqualSmall(pp[a], p3[a]));
    }

    constexpr unsigned samples = 1000;

    std::vector<unsigned> counts(A);
    for (unsigned i = 0; i < samples; ++i) {
        const auto a = p.sampleAction();
        BOOST_CHECK(a < A);
        ++counts[a];
    }

    constexpr unsigned margin = 100;

    BOOST_TEST_INFO(p0 * samples - margin << "<=" << counts[0]);
    BOOST_CHECK(p0 * samples - margin <= counts[0]);
    BOOST_TEST_INFO(counts[0] << "<=" << p0 * samples + margin);
    BOOST_CHECK(counts[0] <= p0 * samples + margin);

    BOOST_CHECK(p1 * samples - margin <= counts[1]);
    BOOST_CHECK(counts[1] <= p1 * samples + margin);

    BOOST_CHECK(p2 * samples - margin <= counts[2]);
    BOOST_CHECK(counts[2] <= p2 * samples + margin);
}
