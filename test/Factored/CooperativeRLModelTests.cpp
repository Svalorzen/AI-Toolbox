#define BOOST_TEST_MODULE Factored_MDP_CooperativeRLModel
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Factored/MDP/CooperativeRLModel.hpp>

#include <AIToolbox/Factored/MDP/Environments/SysAdmin.hpp>

namespace ai = AIToolbox;
namespace aif = AIToolbox::Factored;
namespace afm = AIToolbox::Factored::MDP;

BOOST_AUTO_TEST_CASE( construction ) {
    auto model = afm::makeSysAdminBiRing(7, 0.1, 0.2, 0.3, 0.4, 0.2, 0.2, 0.1);

    afm::CooperativeExperience exp(model.getS(), model.getA(), model.getTransitionFunction().nodes);
    afm::CooperativeRLModel rl(exp, 0.9, false);

    const auto & tt = model.getTransitionFunction();
    const auto & t = rl.getTransitionFunction();
    const auto & r = rl.getRewardFunction();

    BOOST_CHECK_EQUAL(rl.getDiscount(), 0.9);
    BOOST_CHECK_EQUAL(ai::veccmp(model.getS(), rl.getS()), 0);
    BOOST_CHECK_EQUAL(ai::veccmp(model.getA(), rl.getA()), 0);

    BOOST_CHECK_EQUAL(tt.nodes.size(), t.nodes.size());
    // Note that the learned reward function has a different format from the
    // original model (vector<vector<Vector>> vs FactoredMatrix2D). This means
    // we cannot compare them directly.
    BOOST_CHECK_EQUAL(tt.nodes.size(), r.size());

    for (size_t i = 0; i < t.nodes.size(); ++i) {
        BOOST_CHECK_EQUAL(ai::veccmp(tt[i].actionTag, t.nodes[i].actionTag), 0);

        BOOST_CHECK_EQUAL(tt[i].nodes.size(), t.nodes[i].nodes.size());
        BOOST_CHECK_EQUAL(tt[i].nodes.size(), r[i].size());

        for (size_t j = 0; j < t.nodes[i].nodes.size(); ++j) {
            const auto & ttn = tt.nodes[i].nodes[j];
            const auto & tn = t.nodes[i].nodes[j];

            BOOST_CHECK_EQUAL(ai::veccmp(ttn.tag, tn.tag), 0);

            BOOST_CHECK_EQUAL(ttn.matrix.rows(), tn.matrix.rows());
            BOOST_CHECK_EQUAL(ttn.matrix.cols(), tn.matrix.cols());

            BOOST_CHECK_EQUAL(ttn.matrix.rows(), r[i][j].size());

            for (int x = 0; x < tn.matrix.rows(); ++x) {
                BOOST_CHECK_EQUAL(tn.matrix(x, 0), 1.0);
                BOOST_CHECK_EQUAL(r[i][j][x], 0);
                for (int y = 1; y < tn.matrix.cols(); ++y)
                    BOOST_CHECK_EQUAL(tn.matrix(x, y), 0.0);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE( syncing ) {
    auto model = afm::makeSysAdminUniRing(3, 0.1, 0.2, 0.3, 0.4, 0.2, 0.2, 0.1);

    afm::CooperativeExperience exp(model.getS(), model.getA(), model.getTransitionFunction().nodes);
    afm::CooperativeRLModel rl1(exp, 0.9, false);

    aif::Rewards rew(6); rew.setZero();

    rew << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
    exp.record({0, 1, 1, 1, 2, 1}, {0, 0, 0}, {1, 1, 1, 2, 2, 0}, rew);
    rew << 0.0, 1.0, 0.0, 1.0, 0.0, 0.0;
    exp.record({0, 1, 1, 1, 2, 1}, {0, 0, 1}, {0, 2, 1, 1, 0, 0}, rew);
    rew << 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
    exp.record({1, 1, 0, 1, 2, 1}, {0, 0, 1}, {1, 2, 0, 1, 0, 0}, rew);

    rl1.sync();
    afm::CooperativeRLModel rl2(exp, 0.9, true);

    const auto & t1 = rl1.getTransitionFunction();
    const auto & r1 = rl1.getRewardFunction();
    const auto & t2 = rl2.getTransitionFunction();
    const auto & r2 = rl2.getRewardFunction();

    auto id = [](const aif::Factors & f){ return aif::toIndex({3, 3}, f); };

    // Status a1
    BOOST_CHECK_EQUAL(t1.nodes[0].nodes[0].matrix(id({0, 2}), 0), 0.5);
    BOOST_CHECK_EQUAL(t1.nodes[0].nodes[0].matrix(id({0, 2}), 1), 0.5);
    BOOST_CHECK_EQUAL(t1.nodes[0].nodes[0].matrix(id({0, 2}), 2), 0.0);
    BOOST_CHECK_EQUAL(r1[0][0][id({0, 2})], 0.0);

    BOOST_CHECK_EQUAL(t1.nodes[0].nodes[0].matrix(id({1, 2}), 0), 0.0);
    BOOST_CHECK_EQUAL(t1.nodes[0].nodes[0].matrix(id({1, 2}), 1), 1.0);
    BOOST_CHECK_EQUAL(t1.nodes[0].nodes[0].matrix(id({1, 2}), 2), 0.0);
    BOOST_CHECK_EQUAL(r1[0][0][id({1, 2})], 0.0);

    // Load a1
    BOOST_CHECK_EQUAL(t1.nodes[1].nodes[0].matrix(id({0, 1}), 0), 0.0);
    BOOST_CHECK_EQUAL(t1.nodes[1].nodes[0].matrix(id({0, 1}), 1), 0.5);
    BOOST_CHECK_EQUAL(t1.nodes[1].nodes[0].matrix(id({0, 1}), 2), 0.5);
    BOOST_CHECK_EQUAL(r1[1][0][id({0, 1})], 0.5);

    BOOST_CHECK_EQUAL(t1.nodes[1].nodes[0].matrix(id({1, 1}), 0), 0.0);
    BOOST_CHECK_EQUAL(t1.nodes[1].nodes[0].matrix(id({1, 1}), 1), 0.0);
    BOOST_CHECK_EQUAL(t1.nodes[1].nodes[0].matrix(id({1, 1}), 2), 1.0);
    BOOST_CHECK_EQUAL(r1[1][0][id({1, 1})], 1.0);

    // Status a2
    BOOST_CHECK_EQUAL(t1.nodes[2].nodes[0].matrix(id({0, 1}), 0), 0.0);
    BOOST_CHECK_EQUAL(t1.nodes[2].nodes[0].matrix(id({0, 1}), 1), 1.0);
    BOOST_CHECK_EQUAL(t1.nodes[2].nodes[0].matrix(id({0, 1}), 2), 0.0);
    BOOST_CHECK_EQUAL(r1[2][0][id({0, 1})], 0.0);

    BOOST_CHECK_EQUAL(t1.nodes[2].nodes[0].matrix(id({1, 0}), 0), 1.0);
    BOOST_CHECK_EQUAL(t1.nodes[2].nodes[0].matrix(id({1, 0}), 1), 0.0);
    BOOST_CHECK_EQUAL(t1.nodes[2].nodes[0].matrix(id({1, 0}), 2), 0.0);
    BOOST_CHECK_EQUAL(r1[2][0][id({1, 0})], 0.0);

    // Load a2
    BOOST_CHECK_EQUAL(t1.nodes[3].nodes[0].matrix(id({1, 1}), 0), 0.0);
    BOOST_CHECK_EQUAL(t1.nodes[3].nodes[0].matrix(id({1, 1}), 1), 0.5);
    BOOST_CHECK_EQUAL(t1.nodes[3].nodes[0].matrix(id({1, 1}), 2), 0.5);
    BOOST_CHECK_EQUAL(r1[3][0][id({1, 1})], 1.0);

    BOOST_CHECK_EQUAL(t1.nodes[3].nodes[0].matrix(id({0, 1}), 0), 0.0);
    BOOST_CHECK_EQUAL(t1.nodes[3].nodes[0].matrix(id({0, 1}), 1), 1.0);
    BOOST_CHECK_EQUAL(t1.nodes[3].nodes[0].matrix(id({0, 1}), 2), 0.0);
    BOOST_CHECK_EQUAL(r1[3][0][id({0, 1})], 0.0);

    // Status a3
    BOOST_CHECK_EQUAL(t1.nodes[4].nodes[0].matrix(id({1, 2}), 0), 0.0);
    BOOST_CHECK_EQUAL(t1.nodes[4].nodes[0].matrix(id({1, 2}), 1), 0.0);
    BOOST_CHECK_EQUAL(t1.nodes[4].nodes[0].matrix(id({1, 2}), 2), 1.0);
    BOOST_CHECK_EQUAL(r1[4][0][id({1, 2})], 0.0);

    BOOST_CHECK_EQUAL(t1.nodes[4].nodes[1].matrix(2, 0), 1.0);
    BOOST_CHECK_EQUAL(t1.nodes[4].nodes[1].matrix(2, 1), 0.0);
    BOOST_CHECK_EQUAL(t1.nodes[4].nodes[1].matrix(2, 2), 0.0);
    BOOST_CHECK_EQUAL(r1[4][1][2], 0.0);

    // Load a3
    BOOST_CHECK_EQUAL(t1.nodes[5].nodes[0].matrix(id({2, 1}), 0), 1.0);
    BOOST_CHECK_EQUAL(t1.nodes[5].nodes[0].matrix(id({2, 1}), 1), 0.0);
    BOOST_CHECK_EQUAL(t1.nodes[5].nodes[0].matrix(id({2, 1}), 2), 0.0);
    BOOST_CHECK_EQUAL(r1[5][0][2], 0.0);

    BOOST_CHECK_EQUAL(t1.nodes[5].nodes[1].matrix(1, 0), 1.0);
    BOOST_CHECK_EQUAL(t1.nodes[5].nodes[1].matrix(1, 1), 0.0);
    BOOST_CHECK_EQUAL(t1.nodes[5].nodes[1].matrix(1, 2), 0.0);
    BOOST_CHECK_EQUAL(r1[5][1][1], 0.0);

    for (size_t i = 0; i < t1.nodes.size(); ++i) {
        for (size_t j = 0; j < t1.nodes[i].nodes.size(); ++j) {
            const auto & t1n = t1.nodes[i].nodes[j];
            const auto & t2n = t2.nodes[i].nodes[j];

            BOOST_CHECK_EQUAL(t1n.matrix, t2n.matrix);
            BOOST_CHECK_EQUAL(r1[i][j], r2[i][j]);
        }
    }
}
