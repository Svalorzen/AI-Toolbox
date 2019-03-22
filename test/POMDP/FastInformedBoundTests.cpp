#define BOOST_TEST_MODULE POMDP_FastInformedBound
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/POMDP/Algorithms/FastInformedBound.hpp>
#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/POMDP/SparseModel.hpp>
#include <AIToolbox/MDP/SparseModel.hpp>
#include <AIToolbox/Utils/Core.hpp>

#include <AIToolbox/POMDP/Environments/TigerProblem.hpp>

BOOST_AUTO_TEST_CASE( horizon1 ) {
    using namespace AIToolbox;
    using namespace AIToolbox::POMDP;

    auto model = makeTigerProblem();
    model.setDiscount(0.95);

    constexpr unsigned horizon = 1000000;
    constexpr double tolerance = 0.001;
    FastInformedBound solver(horizon, tolerance);
    const auto [var, qfun] = solver(model);

    BOOST_CHECK(var < tolerance);

    // The solution values were taken directly from GapMin's code solution for this problem.
    MDP::QFunction solution(model.getS(), model.getA());
    solution <<
        87.1884724559168, -17.1700289937718,  92.8299710062282,
        87.1884724559168,  92.8299710062282, -17.1700289937718;

    for (size_t s = 0; s < model.getS(); ++s)
        for (size_t a = 0; a < model.getS(); ++a)
            BOOST_CHECK(checkEqualGeneral(solution(s, a), qfun(s,a)));
}


