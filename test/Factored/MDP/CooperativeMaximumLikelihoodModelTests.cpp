#define BOOST_TEST_MODULE Factored_MDP_CooperativeMaximumLikelihoodModel
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include "GlobalFixtures.hpp"

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Factored/MDP/CooperativeMaximumLikelihoodModel.hpp>

#include <AIToolbox/Factored/MDP/Environments/SysAdmin.hpp>

namespace ai = AIToolbox;
namespace aif = AIToolbox::Factored;
namespace afm = AIToolbox::Factored::MDP;

BOOST_AUTO_TEST_CASE( construction ) {
    auto model = afm::makeSysAdminBiRing(7, 0.1, 0.2, 0.3, 0.4, 0.2, 0.2, 0.1);

    afm::CooperativeExperience exp(model.getGraph());
    afm::CooperativeMaximumLikelihoodModel rl(exp, 0.9, false);

    const auto & tt = model.getTransitionFunction().transitions;
    const auto & t = rl.getTransitionFunction().transitions;
    const auto & r = rl.getRewardFunction();

    BOOST_CHECK_EQUAL(rl.getDiscount(), 0.9);
    BOOST_CHECK(ai::veccmp(model.getS(), rl.getS()) == 0);
    BOOST_CHECK(ai::veccmp(model.getA(), rl.getA()) == 0);

    BOOST_CHECK_EQUAL(tt.size(), t.size());
    // Note that the learned reward function has a different format from the
    // original model (vector<Vector> vs FactoredMatrix2D). This means we
    // cannot compare them directly.
    BOOST_CHECK_EQUAL(tt.size(), r.size());
    BOOST_CHECK_EQUAL(&model.getGraph(), &rl.getGraph());

    for (size_t i = 0; i < t.size(); ++i) {
        BOOST_CHECK_EQUAL(tt[i].rows(), t[i].rows());
        BOOST_CHECK_EQUAL(tt[i].cols(), t[i].cols());
        BOOST_CHECK_EQUAL(tt[i].rows(), r[i].size());

        BOOST_CHECK(t[i].col(0).isOnes());
        BOOST_CHECK(t[i].rightCols(t[i].cols() - 1).isZero());
        BOOST_CHECK(r[i].isZero());
    }
}

BOOST_AUTO_TEST_CASE( syncing ) {
    auto model = afm::makeSysAdminUniRing(3, 0.1, 0.2, 0.3, 0.4, 0.2, 0.2, 0.1);

    afm::CooperativeExperience exp(model.getGraph());
    afm::CooperativeMaximumLikelihoodModel rl1(exp, 0.9, false);

    aif::Rewards rew(6); rew.setZero();

    rew << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
    exp.record({0, 1, 1, 1, 2, 1}, {0, 0, 0}, {1, 1, 1, 2, 2, 0}, rew);
    rew << 0.0, 1.0, 0.0, 1.0, 0.0, 0.0;
    exp.record({0, 1, 1, 1, 2, 1}, {0, 0, 1}, {0, 2, 1, 1, 0, 0}, rew);
    rew << 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
    exp.record({1, 1, 0, 1, 2, 1}, {0, 0, 1}, {1, 2, 0, 1, 0, 0}, rew);

    rl1.sync();
    afm::CooperativeMaximumLikelihoodModel rl2(exp, 0.9, true);

    const auto & t1 = rl1.getTransitionFunction().transitions;
    const auto & r1 = rl1.getRewardFunction();
    const auto & t2 = rl2.getTransitionFunction().transitions;
    const auto & r2 = rl2.getRewardFunction();

    // For the uniring, each matrix for each node looks like this:
    //
    //                 3 = S'[i]
    //
    //            ------------------
    //            |     |     |    |
    //            ------------------
    //            |     |     |    |
    //            |     |     |    |
    //  9 = 3*3           ...
    //  (A = 0)   |     |     |    |
    //            |     |     |    |
    //            ------------------
    //            |     |     |    |
    //     3              ...
    //  (A = 1)   |     |     |    |
    //            ------------------
    //
    // Here we look at both A = 0, and A = 1. We create two separate functions
    // to compute the respective ids.

    auto id0 = [](const aif::Factors & f){ return aif::toIndex({3, 3}, f); };
    auto id1 = [](const size_t f){ return 9 + f; };

    // Status a1
    BOOST_CHECK_EQUAL(t1[0](id0({0, 2}), 0), 0.5);
    BOOST_CHECK_EQUAL(t1[0](id0({0, 2}), 1), 0.5);
    BOOST_CHECK_EQUAL(t1[0](id0({0, 2}), 2), 0.0);
    BOOST_CHECK_EQUAL(r1[0][id0({0, 2})], 0.0);

    BOOST_CHECK_EQUAL(t1[0](id0({1, 2}), 0), 0.0);
    BOOST_CHECK_EQUAL(t1[0](id0({1, 2}), 1), 1.0);
    BOOST_CHECK_EQUAL(t1[0](id0({1, 2}), 2), 0.0);
    BOOST_CHECK_EQUAL(r1[0][id0({1, 2})], 0.0);

    // Load a1
    BOOST_CHECK_EQUAL(t1[1](id0({0, 1}), 0), 0.0);
    BOOST_CHECK_EQUAL(t1[1](id0({0, 1}), 1), 0.5);
    BOOST_CHECK_EQUAL(t1[1](id0({0, 1}), 2), 0.5);
    BOOST_CHECK_EQUAL(r1[1][id0({0, 1})], 0.5);

    BOOST_CHECK_EQUAL(t1[1](id0({1, 1}), 0), 0.0);
    BOOST_CHECK_EQUAL(t1[1](id0({1, 1}), 1), 0.0);
    BOOST_CHECK_EQUAL(t1[1](id0({1, 1}), 2), 1.0);
    BOOST_CHECK_EQUAL(r1[1][id0({1, 1})], 1.0);

    // Status a2
    BOOST_CHECK_EQUAL(t1[2](id0({0, 1}), 0), 0.0);
    BOOST_CHECK_EQUAL(t1[2](id0({0, 1}), 1), 1.0);
    BOOST_CHECK_EQUAL(t1[2](id0({0, 1}), 2), 0.0);
    BOOST_CHECK_EQUAL(r1[2][id0({0, 1})], 0.0);

    BOOST_CHECK_EQUAL(t1[2](id0({1, 0}), 0), 1.0);
    BOOST_CHECK_EQUAL(t1[2](id0({1, 0}), 1), 0.0);
    BOOST_CHECK_EQUAL(t1[2](id0({1, 0}), 2), 0.0);
    BOOST_CHECK_EQUAL(r1[2][id0({1, 0})], 0.0);

    // Load a2
    BOOST_CHECK_EQUAL(t1[3](id0({1, 1}), 0), 0.0);
    BOOST_CHECK_EQUAL(t1[3](id0({1, 1}), 1), 0.5);
    BOOST_CHECK_EQUAL(t1[3](id0({1, 1}), 2), 0.5);
    BOOST_CHECK_EQUAL(r1[3][id0({1, 1})], 1.0);

    BOOST_CHECK_EQUAL(t1[3](id0({0, 1}), 0), 0.0);
    BOOST_CHECK_EQUAL(t1[3](id0({0, 1}), 1), 1.0);
    BOOST_CHECK_EQUAL(t1[3](id0({0, 1}), 2), 0.0);
    BOOST_CHECK_EQUAL(r1[3][id0({0, 1})], 0.0);

    // Status a3
    BOOST_CHECK_EQUAL(t1[4](id0({1, 2}), 0), 0.0);
    BOOST_CHECK_EQUAL(t1[4](id0({1, 2}), 1), 0.0);
    BOOST_CHECK_EQUAL(t1[4](id0({1, 2}), 2), 1.0);
    BOOST_CHECK_EQUAL(r1[4][id0({1, 2})], 0.0);

    BOOST_CHECK_EQUAL(t1[4](id1(2), 0), 1.0);
    BOOST_CHECK_EQUAL(t1[4](id1(2), 1), 0.0);
    BOOST_CHECK_EQUAL(t1[4](id1(2), 2), 0.0);
    BOOST_CHECK_EQUAL(r1[4][id1(2)], 0.0);

    // Load a3
    BOOST_CHECK_EQUAL(t1[5](id0({2, 1}), 0), 1.0);
    BOOST_CHECK_EQUAL(t1[5](id0({2, 1}), 1), 0.0);
    BOOST_CHECK_EQUAL(t1[5](id0({2, 1}), 2), 0.0);
    BOOST_CHECK_EQUAL(r1[5][id0({2,1})], 0.0);

    BOOST_CHECK_EQUAL(t1[5](id1(1), 0), 1.0);
    BOOST_CHECK_EQUAL(t1[5](id1(1), 1), 0.0);
    BOOST_CHECK_EQUAL(t1[5](id1(1), 2), 0.0);
    BOOST_CHECK_EQUAL(r1[5][id1(1)], 0.0);

    for (size_t i = 0; i < t1.size(); ++i) {
        BOOST_CHECK_EQUAL(t1[i], t2[i]);
        BOOST_CHECK_EQUAL(r1[i], r2[i]);
    }
}
