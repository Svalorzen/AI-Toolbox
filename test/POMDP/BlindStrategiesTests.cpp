#define BOOST_TEST_MODULE POMDP_BlindStrategies
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/POMDP/Algorithms/BlindStrategies.hpp>
#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/POMDP/SparseModel.hpp>
#include <AIToolbox/MDP/SparseModel.hpp>
#include <AIToolbox/Utils/Core.hpp>

#include <AIToolbox/POMDP/Environments/TigerProblem.hpp>

BOOST_AUTO_TEST_CASE( horizon1 ) {
    using namespace AIToolbox;
    using namespace AIToolbox::POMDP;
    using namespace TigerProblemEnums;

    auto model = makeTigerProblem();
    model.setDiscount(0.95);

    constexpr unsigned horizon = 1;
    BlindStrategies solver(horizon, 0.1);
    const auto [var, vlist] = solver(model, false);

    BOOST_CHECK_EQUAL(var, 42.75);

    BOOST_CHECK_EQUAL(vlist[A_LISTEN].values[TIG_LEFT], -1.95);
    BOOST_CHECK_EQUAL(vlist[A_LISTEN].values[TIG_RIGHT], -1.95);

    BOOST_CHECK_EQUAL(vlist[A_LEFT].values[TIG_LEFT],  -100.0 + 0.95 * (0.5 * 10.0 - 0.5 * 100));
    BOOST_CHECK_EQUAL(vlist[A_LEFT].values[TIG_RIGHT],   10.0 + 0.95 * (0.5 * 10.0 - 0.5 * 100));

    BOOST_CHECK_EQUAL(vlist[A_RIGHT].values[TIG_LEFT],   10.0 + 0.95 * (0.5 * 10.0 - 0.5 * 100));
    BOOST_CHECK_EQUAL(vlist[A_RIGHT].values[TIG_RIGHT],-100.0 + 0.95 * (0.5 * 10.0 - 0.5 * 100));
}

BOOST_AUTO_TEST_CASE( infiniteHorizonSpeededUp ) {
    using namespace AIToolbox;
    using namespace AIToolbox::POMDP;
    using namespace TigerProblemEnums;

    constexpr double discount = 0.95;
    auto model = makeTigerProblem();
    model.setDiscount(discount);

    constexpr unsigned horizon = 100000;
    constexpr double tolerance = 0.0001;
    BlindStrategies solver(horizon, tolerance);

    const auto [varSpeed, vlistSpeed] = solver(model, true);
    const auto [varNormal, vlistNormal] = solver(model, false);

    BOOST_CHECK(varSpeed < solver.getTolerance());
    BOOST_CHECK(varNormal < solver.getTolerance());

    // The following bounds hold because there exist a bound on how much the
    // error is over the final V* (which in this case is not optimal but it's
    // what we're converging to). The bound can be summarized as:
    //
    // | V*(s) - V(s) | <= eps / (1 - discount)
    //
    // Since here we have two solutions, each of which may be far away from the
    // other, we simply double the bound range and then it must hold.

    BOOST_CHECK(std::abs(
        vlistSpeed[A_LISTEN].values[TIG_LEFT] -
        vlistNormal[A_LISTEN].values[TIG_LEFT]
    ) <= (2 * tolerance) / (1 - discount));
    BOOST_CHECK(std::abs(
        vlistSpeed[A_LISTEN].values[TIG_RIGHT] -
        vlistNormal[A_LISTEN].values[TIG_RIGHT]
    ) <= (2 * tolerance) / (1 - discount));

    BOOST_CHECK(std::abs(
        vlistSpeed[A_LEFT].values[TIG_LEFT] -
        vlistNormal[A_LEFT].values[TIG_LEFT]
    ) <= (2 * tolerance) / (1 - discount));
    BOOST_CHECK(std::abs(
        vlistSpeed[A_LEFT].values[TIG_RIGHT] -
        vlistNormal[A_LEFT].values[TIG_RIGHT]
    ) <= (2 * tolerance) / (1 - discount));

    BOOST_CHECK(std::abs(
        vlistSpeed[A_RIGHT].values[TIG_LEFT] -
        vlistNormal[A_RIGHT].values[TIG_LEFT]
    ) <= (2 * tolerance) / (1 - discount));
    BOOST_CHECK(std::abs(
        vlistSpeed[A_RIGHT].values[TIG_RIGHT] -
        vlistNormal[A_RIGHT].values[TIG_RIGHT]
    ) <= (2 * tolerance) / (1 - discount));
}
