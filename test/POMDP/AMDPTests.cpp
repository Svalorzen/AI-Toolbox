#define BOOST_TEST_MODULE POMDP_AMDP
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/MDP/Algorithms/ValueIteration.hpp>

#include <AIToolbox/POMDP/Algorithms/AMDP.hpp>
#include <AIToolbox/POMDP/Algorithms/IncrementalPruning.hpp>

#include <AIToolbox/POMDP/Policies/Policy.hpp>
#include <AIToolbox/MDP/Policies/QGreedyPolicy.hpp>

#include <AIToolbox/POMDP/SparseModel.hpp>
#include <AIToolbox/MDP/SparseModel.hpp>
#include <AIToolbox/POMDP/Types.hpp>

#include <AIToolbox/POMDP/Environments/TigerProblem.hpp>

BOOST_AUTO_TEST_CASE( discountedHorizon ) {
    using namespace AIToolbox;
    using namespace AIToolbox::POMDP;

    auto model = makeTigerProblem();
    model.setDiscount(0.95);

    unsigned horizon = 4;
    IncrementalPruning ipsolver(horizon, 0.0);

    auto truth = ipsolver(model);
    Policy truthPolicy(model.getS(), model.getA(), model.getO(), std::get<1>(truth));

    AMDP converter(4000, 70);
    const auto [simplerModel, beliefConverter] = converter.discretizeDense(model);

    MDP::ValueIteration solver(horizon);

    auto solution = solver(simplerModel);
    MDP::QGreedyPolicy policy(std::get<2>(solution));

    // NOTE: This test is very very fragile, the point is that AMDP is
    // by definition a very approximate method of solving a POMDP, so
    // I'm not exactly sure what would be the best way to consistently
    // test its behaviour. For now, this will have to do. Trying a number
    // of random beliefs works in the majority of cases but for some
    // particular beliefs tends to fail (possibly near the edges of an
    // entropy bucket), so that's not really doable.
    Matrix2D beliefs(5, 2);
    beliefs << 0.5,     0.5,
               1.0,     0.0,
               0.25,    0.75,
               0.98,    0.02,
               0.33,    0.66;

    for ( auto i = 0; i < beliefs.rows(); ++i )
        BOOST_CHECK_EQUAL( truthPolicy.sampleAction(beliefs.row(i)), policy.sampleAction( beliefConverter(beliefs.row(i)) ) );
}

BOOST_AUTO_TEST_CASE( discountedHorizonSparse ) {
    using namespace AIToolbox;
    using namespace AIToolbox::POMDP;

    auto model = makeTigerProblem();
    SparseModel<MDP::SparseModel> sparseModel = model;
    model.setDiscount(0.95);

    unsigned horizon = 4;
    IncrementalPruning ipsolver(horizon, 0.0);

    auto truth = ipsolver(model);
    Policy truthPolicy(model.getS(), model.getA(), model.getO(), std::get<1>(truth));

    AMDP converter(4000, 70);
    const auto [simplerModel, beliefConverter] = converter.discretizeSparse(sparseModel);

    MDP::ValueIteration solver(horizon);

    auto solution = solver(simplerModel);
    MDP::QGreedyPolicy policy(std::get<2>(solution));

    // NOTE: This test is very very fragile, the point is that AMDP is
    // by definition a very approximate method of solving a POMDP, so
    // I'm not exactly sure what would be the best way to consistently
    // test its behaviour. For now, this will have to do. Trying a number
    // of random beliefs works in the majority of cases but for some
    // particular beliefs tends to fail (possibly near the edges of an
    // entropy bucket), so that's not really doable.
    Matrix2D beliefs(5, 2);
    beliefs << 0.5,     0.5,
               1.0,     0.0,
               0.25,    0.75,
               0.98,    0.02,
               0.33,    0.66;

    for ( auto i = 0; i < beliefs.rows(); ++i )
        BOOST_CHECK_EQUAL( truthPolicy.sampleAction(beliefs.row(i)), policy.sampleAction( beliefConverter(beliefs.row(i)) ) );
}
