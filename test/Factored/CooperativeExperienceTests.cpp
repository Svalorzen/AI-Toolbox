#define BOOST_TEST_MODULE Factored_MDP_CooperativeExperience
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Factored/MDP/CooperativeExperience.hpp>
#include <AIToolbox/Factored/MDP/CooperativeModel.hpp>

#include <AIToolbox/Factored/MDP/Environments/SysAdmin.hpp>

namespace ai = AIToolbox;
namespace aif = AIToolbox::Factored;
namespace afm = AIToolbox::Factored::MDP;

BOOST_AUTO_TEST_CASE( construction ) {
    auto model = afm::makeSysAdminBiRing(7, 0.1, 0.2, 0.3, 0.4, 0.2, 0.2, 0.1);

    afm::CooperativeExperience exp(model.getS(), model.getA(), model.getTransitionFunction().nodes);

    const auto & t = model.getTransitionFunction();
    const auto & v = exp.getVisitTable();
    const auto & r = exp.getRewardMatrix();

    const auto & S = model.getS();
    const auto & A = model.getA();

    BOOST_CHECK_EQUAL(ai::veccmp(S, exp.getS()), 0);
    BOOST_CHECK_EQUAL(ai::veccmp(A, exp.getA()), 0);

    BOOST_CHECK_EQUAL(r.size(), t.nodes.size());
    BOOST_CHECK_EQUAL(r.size(), v.size());

    for (size_t i = 0; i < t.nodes.size(); ++i) {
        BOOST_CHECK_EQUAL(ai::veccmp(r[i].actionTag, t.nodes[i].actionTag), 0);

        BOOST_CHECK_EQUAL(r[i].nodes.size(), t.nodes[i].nodes.size());
        BOOST_CHECK_EQUAL(r[i].nodes.size(), v[i].size());

        for (size_t j = 0; j < t.nodes[i].nodes.size(); ++j) {
            const auto & tn = t.nodes[i].nodes[j];
            const auto & rn = r[i].nodes[j];

            BOOST_CHECK_EQUAL(ai::veccmp(rn.tag, tn.tag), 0);

            BOOST_CHECK_EQUAL(rn.matrix.rows(), tn.matrix.rows());
            BOOST_CHECK_EQUAL(rn.matrix.cols(), tn.matrix.cols() + 1);

            BOOST_CHECK_EQUAL(rn.matrix.rows(), v[i][j].rows());
            BOOST_CHECK_EQUAL(rn.matrix.cols(), v[i][j].cols());

            for (int x = 0; x < rn.matrix.rows(); ++x) {
                for (int y = 0; y < rn.matrix.cols(); ++y) {
                    BOOST_CHECK_EQUAL(rn.matrix(x, y), 0.0);
                    BOOST_CHECK_EQUAL(v[i][j](x, y), 0);
                }
            }
        }
    }
}

BOOST_AUTO_TEST_CASE( recording ) {
    auto model = afm::makeSysAdminUniRing(7, 0.1, 0.2, 0.3, 0.4, 0.2, 0.2, 0.1);

    afm::CooperativeExperience exp(model.getS(), model.getA(), model.getTransitionFunction().nodes);

    const auto & S = model.getS();

    ai::Vector rew(S.size()); rew.setZero();
    aif::State a {0, 0, 0, 0, 0, 0, 0};

    //            0  1  2  3  4  5  6  7  8  9  A  B  C  D
    //            s  l  s  l  s  l  s  l  s  l  s  l  s  l
    aif::State s {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2};
    aif::State s1{0, 1, 1, 0, 1, 1, 1, 2, 2, 1, 2, 2, 2, 0};
    rew <<        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0;

    const auto & indeces = exp.record(s, a, s1, rew);

    auto id = [](const aif::Factors & f){ return aif::toIndex({3, 3}, f); };

    afm::CooperativeExperience::Indeces indexSolution{
        {0, id({0,2})},
        {0, id({0,0})},
        {0, id({0,0})},
        {0, id({0,0})},
        {0, id({0,0})},
        {0, id({0,0})},
        {0, id({0,1})},
        {0, id({1,1})},
        {0, id({1,1})},
        {0, id({1,1})},
        {0, id({1,1})},
        {0, id({1,1})},
        {0, id({1,2})},
        {0, id({2,2})},
    };

    BOOST_CHECK_EQUAL(indeces.size(), indexSolution.size());
    for (size_t i = 0; i < indeces.size(); ++i) {
        BOOST_CHECK_EQUAL(indeces[i].first, indexSolution[i].first);
        BOOST_CHECK_EQUAL(indeces[i].second, indexSolution[i].second);
    }

    // status elements depend on them + status previous agent.
    // load   elements depend on them + status same agent.

    struct Solution {
        size_t parent;
        std::vector<std::tuple<size_t, size_t, double>> data;
        size_t visitSum;
        double rewardSum;
    };

    std::vector<std::vector<Solution>> solutions;
    solutions.resize(S.size());

    //                                   s values   s1      r       sums
    //                                      |        |      |
    //                                      v        v      v
    solutions[0] .emplace_back(Solution{id({0,2}), {{0, 1, 0.0}},  1, 0.0});
    solutions[1] .emplace_back(Solution{id({0,0}), {{1, 1, 0.0}},  1, 0.0});
    solutions[2] .emplace_back(Solution{id({0,0}), {{1, 1, 0.0}},  1, 0.0});
    solutions[3] .emplace_back(Solution{id({0,0}), {{0, 1, 0.0}},  1, 0.0});
    solutions[4] .emplace_back(Solution{id({0,0}), {{1, 1, 0.0}},  1, 0.0});
    solutions[5] .emplace_back(Solution{id({0,0}), {{1, 1, 0.0}},  1, 0.0});
    solutions[6] .emplace_back(Solution{id({0,1}), {{1, 1, 0.0}},  1, 0.0});
    solutions[7] .emplace_back(Solution{id({1,1}), {{2, 1, 1.0}},  1, 1.0});
    solutions[8] .emplace_back(Solution{id({1,1}), {{2, 1, 0.0}},  1, 0.0});
    solutions[9] .emplace_back(Solution{id({1,1}), {{1, 1, 0.0}},  1, 0.0});
    solutions[10].emplace_back(Solution{id({1,1}), {{2, 1, 0.0}},  1, 0.0});
    solutions[11].emplace_back(Solution{id({1,1}), {{2, 1, 1.0}},  1, 1.0});
    solutions[12].emplace_back(Solution{id({1,2}), {{2, 1, 0.0}},  1, 0.0});
    solutions[13].emplace_back(Solution{id({2,2}), {{0, 1, 0.0}},  1, 0.0});

    const auto & v = exp.getVisitTable();
    const auto & r = exp.getRewardMatrix();

    // state id, action id, parent id, s'
    for (size_t i = 0; i < solutions.size(); ++i) {
        for (size_t x = 0; x < 9; ++x) {
            const Solution * ps = nullptr;
            for (const auto & s : solutions[i])
                if (s.parent == x)
                    ps = &s;

            if (ps) {
                BOOST_CHECK_EQUAL(v[i][0](x, 3), ps->visitSum);
                BOOST_CHECK_EQUAL(r[i].nodes[0].matrix(x, 3), ps->rewardSum);
            } else {
                BOOST_CHECK_EQUAL(v[i][0](x, 3), 0);
                BOOST_CHECK_EQUAL(r[i].nodes[0].matrix(x, 3), 0.0);
            }

            for (size_t y = 0; y < 3; ++y) {
                size_t z = 0;
                for (; ps && z < ps->data.size(); ++z)
                    if (std::get<0>(ps->data[z]) == y)
                        break;

                if (ps && z < ps->data.size()) {
                    BOOST_CHECK_EQUAL(v[i][0](x, y),              std::get<1>(ps->data[z]));
                    BOOST_CHECK_EQUAL(r[i].nodes[0].matrix(x, y), std::get<2>(ps->data[z]));
                } else {
                    BOOST_CHECK_EQUAL(v[i][0](x, y), 0);
                    BOOST_CHECK_EQUAL(r[i].nodes[0].matrix(x, y), 0.0);
                }
            }
        }
    }

    // No action 1
    for (size_t s = 0; s < 14; ++s) {
        for (size_t x = 0; x < 3; ++x) {
            for (size_t y = 0; y < 4; ++y) {
                BOOST_CHECK_EQUAL(v[s][1](x, y), 0);
                BOOST_CHECK_EQUAL(r[s].nodes[1].matrix(x, y), 0.0);
            }
        }
    }

    //               0  1  2  3  4  5  6  7  8  9  A  B  C  D
    //               s  l  s  l  s  l  s  l  s  l  s  l  s  l
    // s            {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2};
    // s1           {0, 1, 1, 0, 1, 1, 1, 2, 2, 1, 2, 2, 2, 0};
    aif::State ss = {0, 1, 1, 0, 1, 1, 1, 2, 1, 2, 2, 1, 2, 0};
    rew <<           0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0;

    exp.record(s, a, ss, rew);

    for (auto & s : solutions)
        s.clear();

    //                                   s values   s1      r    ss              sums
    //                                      |        |      |     |
    //                                      v        v      v     v
    solutions[0] .emplace_back(Solution{id({0,2}), {{0, 2, 0.0}},               2, 0.0});
    solutions[1] .emplace_back(Solution{id({0,0}), {{1, 2, 0.0}},               2, 0.0});
    solutions[2] .emplace_back(Solution{id({0,0}), {{1, 2, 0.0}},               2, 0.0});
    solutions[3] .emplace_back(Solution{id({0,0}), {{0, 2, 0.0}},               2, 0.0});
    solutions[4] .emplace_back(Solution{id({0,0}), {{1, 2, 0.0}},               2, 0.0});
    solutions[5] .emplace_back(Solution{id({0,0}), {{1, 2, 0.0}},               2, 0.0});
    solutions[6] .emplace_back(Solution{id({0,1}), {{1, 2, 0.0}},               2, 0.0});
    solutions[7] .emplace_back(Solution{id({1,1}), {{2, 2, 2.0}},               2, 2.0});
    solutions[8] .emplace_back(Solution{id({1,1}), {{2, 1, 0.0}, {1, 1, 0.0}},  2, 0.0});
    solutions[9] .emplace_back(Solution{id({1,1}), {{1, 1, 0.0}, {2, 1, 1.0}},  2, 1.0});
    solutions[10].emplace_back(Solution{id({1,1}), {{2, 2, 0.0}},               2, 0.0});
    solutions[11].emplace_back(Solution{id({1,1}), {{2, 1, 1.0}, {1, 1, 0.0}},  2, 1.0});
    solutions[12].emplace_back(Solution{id({1,2}), {{2, 2, 0.0}},               2, 0.0});
    solutions[13].emplace_back(Solution{id({2,2}), {{0, 2, 0.0}},               2, 0.0});

    // state id, action id, parent id, s'
    for (size_t i = 0; i < solutions.size(); ++i) {
        for (size_t x = 0; x < 9; ++x) {
            const Solution * ps = nullptr;
            for (const auto & s : solutions[i])
                if (s.parent == x)
                    ps = &s;

            if (ps) {
                BOOST_CHECK_EQUAL(v[i][0](x, 3), ps->visitSum);
                BOOST_CHECK_EQUAL(r[i].nodes[0].matrix(x, 3), ps->rewardSum);
            } else {
                BOOST_CHECK_EQUAL(v[i][0](x, 3), 0);
                BOOST_CHECK_EQUAL(r[i].nodes[0].matrix(x, 3), 0.0);
            }

            for (size_t y = 0; y < 3; ++y) {
                size_t z = 0;
                for (; ps && z < ps->data.size(); ++z)
                    if (std::get<0>(ps->data[z]) == y)
                        break;

                if (ps && z < ps->data.size()) {
                    BOOST_CHECK_EQUAL(v[i][0](x, y),              std::get<1>(ps->data[z]));
                    BOOST_CHECK_EQUAL(r[i].nodes[0].matrix(x, y), std::get<2>(ps->data[z]));
                } else {
                    BOOST_CHECK_EQUAL(v[i][0](x, y), 0);
                    BOOST_CHECK_EQUAL(r[i].nodes[0].matrix(x, y), 0.0);
                }
            }
        }
    }

    // No action 1
    for (size_t s = 0; s < 14; ++s) {
        for (size_t x = 0; x < 3; ++x) {
            for (size_t y = 0; y < 4; ++y) {
                BOOST_CHECK_EQUAL(v[s][1](x, y), 0);
                BOOST_CHECK_EQUAL(r[s].nodes[1].matrix(x, y), 0.0);
            }
        }
    }
}
