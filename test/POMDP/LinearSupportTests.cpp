#define BOOST_TEST_MODULE POMDP_Incremental_Pruning
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/POMDP/Algorithms/LinearSupport.hpp>
#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/POMDP/SparseModel.hpp>
#include <AIToolbox/MDP/SparseModel.hpp>
#include <AIToolbox/Utils/Core.hpp>

#include "Utils/TigerProblem.hpp"

BOOST_AUTO_TEST_CASE( linear_support ) {
}

BOOST_AUTO_TEST_CASE( optimistic_value_discovery ) {
    using namespace AIToolbox;

    std::vector<std::pair<Vector, double>> beliefs = {
        {(Vector(3) << 1.0, 0.0, 0.0).finished(), 10.0},
        {(Vector(3) << 0.0, 1.0, 0.0).finished(), 5.0},
        {(Vector(3) << 0.0, 0.0, 1.0).finished(), -10.0},
    };

    Vector b(3);
    b.fill(1.0/3.0);

    constexpr double solution = (10.0 + 5.0 - 10.0) / 3.0;

    const auto v = POMDP::computeOptimisticValue(b, std::begin(beliefs), std::end(beliefs));

    BOOST_CHECK(checkEqualGeneral(v, solution));
}
