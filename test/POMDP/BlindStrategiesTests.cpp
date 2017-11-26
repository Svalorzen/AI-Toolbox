#define BOOST_TEST_MODULE POMDP_BlindStrategies
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/POMDP/Algorithms/BlindStrategies.hpp>
#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/POMDP/SparseModel.hpp>
#include <AIToolbox/MDP/SparseModel.hpp>
#include <AIToolbox/Utils/Core.hpp>

#include "Utils/TigerProblem.hpp"

BOOST_AUTO_TEST_CASE( horizon1 ) {
    using namespace AIToolbox;

    auto model = makeTigerProblem();
    model.setDiscount(0.95);

    constexpr unsigned horizon = 1;
    POMDP::BlindStrategies solver(horizon, 0.1);
    const auto [var, vlist] = solver(model, false);

    BOOST_CHECK_EQUAL(var, 42.75);

    BOOST_CHECK_EQUAL(std::get<POMDP::VALUES>(vlist[A_LISTEN])[TIG_LEFT], -1.95);
    BOOST_CHECK_EQUAL(std::get<POMDP::VALUES>(vlist[A_LISTEN])[TIG_RIGHT], -1.95);

    BOOST_CHECK_EQUAL(std::get<POMDP::VALUES>(vlist[A_LEFT])[TIG_LEFT],  -100.0 + 0.95 * (0.5 * 10.0 - 0.5 * 100));
    BOOST_CHECK_EQUAL(std::get<POMDP::VALUES>(vlist[A_LEFT])[TIG_RIGHT],   10.0 + 0.95 * (0.5 * 10.0 - 0.5 * 100));

    BOOST_CHECK_EQUAL(std::get<POMDP::VALUES>(vlist[A_RIGHT])[TIG_LEFT],   10.0 + 0.95 * (0.5 * 10.0 - 0.5 * 100));
    BOOST_CHECK_EQUAL(std::get<POMDP::VALUES>(vlist[A_RIGHT])[TIG_RIGHT],-100.0 + 0.95 * (0.5 * 10.0 - 0.5 * 100));
}
