#define BOOST_TEST_MODULE UtilsAdam
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Types.hpp>
#include <AIToolbox/Utils/Adam.hpp>

namespace ai = AIToolbox;

double objective(const ai::Vector & p) {
    return p.squaredNorm();
}

void derivative(const ai::Vector & p, ai::Vector & grad) {
    grad[0] = 2 * p[0];
    grad[1] = 2 * p[1];
}

BOOST_AUTO_TEST_CASE( simple_gradient_descent ) {
    using namespace AIToolbox;

    ai::Vector point(2);
    point << -0.21, 0.47;

    ai::Vector gradient(2);
    derivative(point, gradient);

    ai::Adam adam(&point, &gradient, 0.02);

    for (auto i = 0; i < 100; ++i) {
        adam.step();
        derivative(point, gradient);
    }

    double val = objective(point);
    BOOST_TEST_INFO(val);
    BOOST_CHECK(val < 1e-5);
}
