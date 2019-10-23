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

    afm::CooperativeExperience exp(model.getGraph());

    const auto & t = model.getTransitionFunction();
    const auto & tt = t.transitions;
    const auto & v = exp.getVisitTable();
    const auto & r = exp.getRewardMatrix();
    const auto & m = exp.getM2Matrix();

    const auto & S = model.getS();
    const auto & A = model.getA();

    BOOST_CHECK_EQUAL(ai::veccmp(S, exp.getS()), 0);
    BOOST_CHECK_EQUAL(ai::veccmp(A, exp.getA()), 0);

    BOOST_CHECK_EQUAL(r.size(), S.size());
    BOOST_CHECK_EQUAL(r.size(), tt.size());
    BOOST_CHECK_EQUAL(r.size(), v.size());

    for (size_t i = 0; i < S.size(); ++i) {
        BOOST_CHECK_EQUAL(r[i].rows(), tt[i].rows());
        BOOST_CHECK_EQUAL(r[i].rows(), v[i].rows());

        BOOST_CHECK_EQUAL(r[i].cols(), 1);
        BOOST_CHECK_EQUAL(m[i].cols(), 1);

        BOOST_CHECK(r[i].isZero());
        BOOST_CHECK(v[i].isZero());
    }
}

BOOST_AUTO_TEST_CASE( recording ) {
    auto model = afm::makeSysAdminUniRing(7, 0.1, 0.2, 0.3, 0.4, 0.2, 0.2, 0.1);

    afm::CooperativeExperience exp(model.getGraph());

    const auto & S = model.getS();

    ai::Vector rew(S.size()); rew.setZero();
    aif::State a {0, 0, 0, 0, 0, 0, 0};

    //            0  1  2  3  4  5  6  7  8  9  A  B  C  D
    //            s  l  s  l  s  l  s  l  s  l  s  l  s  l
    aif::State s {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2};
    aif::State s1{0, 1, 1, 0, 1, 1, 1, 2, 2, 1, 2, 2, 2, 0};
    rew <<        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0;

    const auto & indeces = exp.record(s, a, s1, rew);

    // Pre-compute the indeces that should get updated here.
    //
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
    // Since we are only looking at A = 0, to compute the indeces we can simply
    // take the toIndex result on the value of the two parents of each S'
    // element.

    auto id = [](const aif::Factors & f){ return aif::toIndex({3, 3}, f); };

    // status elements depend on them + status previous agent.
    // load   elements depend on them + status same agent.
    afm::CooperativeExperience::Indeces indexSolution {
    //      s s_dep (status->prev status, load->status)
    //      | |
    //      v v
        id({0,2}), // 0, C
        id({0,0}), // 0, 1
        id({0,0}), // ...
        id({0,0}),
        id({0,0}),
        id({0,0}),
        id({0,1}), // 4, 6
        id({1,1}),
        id({1,1}),
        id({1,1}),
        id({1,1}),
        id({1,1}),
        id({1,2}), // A, C
        id({2,2}),
    };

    BOOST_CHECK_EQUAL(indeces.size(), indexSolution.size());
    for (size_t i = 0; i < indeces.size(); ++i) {
        BOOST_TEST_INFO(i);
        BOOST_CHECK_EQUAL(indeces[i], indexSolution[i]);
    }

    struct Solution {
        // Id of the updated row
        size_t parent;
        //                   s1 value, visits
        std::vector<std::tuple<size_t, size_t>> data;
        size_t visitSum;
        double rewardAvg;
    };

    std::vector<Solution> solutions;

    //                               s values   s1 vis vsums ravg
    //                                  |        |  |
    //                                  v        v  v
    solutions.emplace_back(Solution{id({0,2}), {{0, 1}},  1, 0.0});
    solutions.emplace_back(Solution{id({0,0}), {{1, 1}},  1, 0.0});
    solutions.emplace_back(Solution{id({0,0}), {{1, 1}},  1, 0.0});
    solutions.emplace_back(Solution{id({0,0}), {{0, 1}},  1, 0.0});
    solutions.emplace_back(Solution{id({0,0}), {{1, 1}},  1, 0.0});
    solutions.emplace_back(Solution{id({0,0}), {{1, 1}},  1, 0.0});
    solutions.emplace_back(Solution{id({0,1}), {{1, 1}},  1, 0.0});
    solutions.emplace_back(Solution{id({1,1}), {{2, 1}},  1, 1.0});
    solutions.emplace_back(Solution{id({1,1}), {{2, 1}},  1, 0.0});
    solutions.emplace_back(Solution{id({1,1}), {{1, 1}},  1, 0.0});
    solutions.emplace_back(Solution{id({1,1}), {{2, 1}},  1, 0.0});
    solutions.emplace_back(Solution{id({1,1}), {{2, 1}},  1, 1.0});
    solutions.emplace_back(Solution{id({1,2}), {{2, 1}},  1, 0.0});
    solutions.emplace_back(Solution{id({2,2}), {{0, 1}},  1, 0.0});

    const auto & v = exp.getVisitTable();
    const auto & r = exp.getRewardMatrix();

    // For each state features
    for (size_t i = 0; i < S.size(); ++i) {
        // For entries referring to action 0
        for (size_t j = 0; j < model.getGraph().getPartialSize(i, 0); ++j) {
            // Check whether we should have updated something in this line
            const Solution * ps = nullptr;
            if (solutions[i].parent == j)
                ps = &solutions[i];

            // If we have experience for this, check that it matches.
            // Otherwise, check that everything is still zero.
            //
            // Here we check the sums.
            if (ps) {
                BOOST_CHECK_EQUAL(v[i](j, 3), ps->visitSum);
                BOOST_CHECK_EQUAL(r[i][j], ps->rewardAvg);
            } else {
                BOOST_CHECK_EQUAL(v[i](j, 3), 0);
                BOOST_CHECK_EQUAL(r[i][j], 0.0);
            }

            // Here we check individual values.
            for (size_t y = 0; y < 3; ++y) {
                size_t z = 0;
                // See if we can find a data entry that matches this s' value.
                for (; ps && z < ps->data.size(); ++z)
                    if (std::get<0>(ps->data[z]) == y)
                        break;

                // If we can, we check against the solution.
                // Otherwise, everything should be empty.
                if (ps && z < ps->data.size()) {
                    BOOST_CHECK_EQUAL(v[i](j, y), std::get<1>(ps->data[z]));
                } else {
                    BOOST_CHECK_EQUAL(v[i](j, y), 0);
                }
            }
        }
    }

    // No action 1
    for (size_t i = 0; i < S.size(); ++i) {
        auto action1Rows = exp.getGraph().getPartialSize(i, 1);

        BOOST_CHECK(v[i].bottomRows(action1Rows).isZero());
        BOOST_CHECK(r[i].bottomRows(action1Rows).isZero());
    }

    //               0  1  2  3  4  5  6  7  8  9  A  B  C  D
    //               s  l  s  l  s  l  s  l  s  l  s  l  s  l
    // s            {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2};
    // s1           {0, 1, 1, 0, 1, 1, 1, 2, 2, 1, 2, 2, 2, 0};
    aif::State ss = {0, 1, 1, 0, 1, 1, 1, 2, 1, 2, 2, 1, 2, 0};
    rew <<           0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0;

    exp.record(s, a, ss, rew);

    solutions.clear();

    //                               s values   s1 vis        vsums ravg
    //                                  |        |  |
    //                                  v        v  v
    solutions.emplace_back(Solution{id({0,2}), {{0, 2}},         2, 0.0});
    solutions.emplace_back(Solution{id({0,0}), {{1, 2}},         2, 0.0});
    solutions.emplace_back(Solution{id({0,0}), {{1, 2}},         2, 0.0});
    solutions.emplace_back(Solution{id({0,0}), {{0, 2}},         2, 0.0});
    solutions.emplace_back(Solution{id({0,0}), {{1, 2}},         2, 0.0});
    solutions.emplace_back(Solution{id({0,0}), {{1, 2}},         2, 0.0});
    solutions.emplace_back(Solution{id({0,1}), {{1, 2}},         2, 0.0});
    solutions.emplace_back(Solution{id({1,1}), {{2, 2}},         2, 2.0/2});
    solutions.emplace_back(Solution{id({1,1}), {{2, 1}, {1, 1}}, 2, 0.0});
    solutions.emplace_back(Solution{id({1,1}), {{1, 1}, {2, 1}}, 2, 1.0/2});
    solutions.emplace_back(Solution{id({1,1}), {{2, 2}},         2, 0.0});
    solutions.emplace_back(Solution{id({1,1}), {{2, 1}, {1, 1}}, 2, 1.0/2});
    solutions.emplace_back(Solution{id({1,2}), {{2, 2}},         2, 0.0});
    solutions.emplace_back(Solution{id({2,2}), {{0, 2}},         2, 0.0});

    // Same as before, with the updated counters.
    for (size_t i = 0; i < solutions.size(); ++i) {
        // For entries referring to action 0
        for (size_t j = 0; j < model.getGraph().getPartialSize(i, 0); ++j) {
            const Solution * ps = nullptr;
            if (solutions[i].parent == j)
                ps = &solutions[i];

            // Sums..
            if (ps) {
                BOOST_CHECK_EQUAL(v[i](j, 3), ps->visitSum);
                BOOST_CHECK_EQUAL(r[i][j], ps->rewardAvg);
            } else {
                BOOST_CHECK_EQUAL(v[i](j, 3), 0);
                BOOST_CHECK_EQUAL(r[i][j], 0.0);
            }

            // Individual entries..
            for (size_t y = 0; y < 3; ++y) {
                size_t z = 0;
                for (; ps && z < ps->data.size(); ++z)
                    if (std::get<0>(ps->data[z]) == y)
                        break;

                if (ps && z < ps->data.size()) {
                    BOOST_CHECK_EQUAL(v[i](j, y), std::get<1>(ps->data[z]));
                } else {
                    BOOST_CHECK_EQUAL(v[i](j, y), 0);
                }
            }
        }
    }

    // No action 1
    for (size_t i = 0; i < S.size(); ++i) {
        auto action1Rows = exp.getGraph().getPartialSize(i, 1);

        BOOST_CHECK(v[i].bottomRows(action1Rows).isZero());
        BOOST_CHECK(r[i].bottomRows(action1Rows).isZero());
    }
}
