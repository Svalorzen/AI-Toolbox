#define BOOST_TEST_MODULE MDP_ValueIteration
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/MDP/Algorithms/Utils/PolicyEvaluation.hpp>
#include <AIToolbox/MDP/Policies/Policy.hpp>
#include <AIToolbox/MDP/Model.hpp>

#include <AIToolbox/MDP/Environments/CornerProblem.hpp>

#include "Utils/OldMDPModel.hpp"

BOOST_AUTO_TEST_CASE( escapeToCorners ) {
    using namespace AIToolbox::MDP;

    GridWorld grid(4, 4);

    // We want the grid to be deterministic
    Model model = makeCornerProblem(grid, 1.0);
    model.setDiscount(1.0);
    size_t S = model.getS(), A = model.getA();

    Policy randomPolicy(S, A);

    // Truth values are taken from the Sutton&Barto book, although they are
    // only accurate to the first significant digit. So we multiply everything
    // by 10 and then round, and then compare.
    auto checkSolution = [S](const Values & truth, const Values & solution) {
        for (size_t s = 0; s < S; ++s)
            BOOST_CHECK(AIToolbox::checkEqualGeneral(std::round(truth[s] * 10.0),
                                                     std::round(solution[s] * 10.0))
            );
    };

    Values truthHorizon1(S);
    truthHorizon1 << 0.0, -1.0, -1.0, -1.0,
                    -1.0, -1.0, -1.0, -1.0,
                    -1.0, -1.0, -1.0, -1.0,
                    -1.0, -1.0, -1.0,  0.0;

    PolicyEvaluation ev(model, 1, 0.0);
    auto solution = ev(randomPolicy);
    checkSolution(truthHorizon1, std::get<1>(solution));

    // NOTE: Here we swap the 1.7s with 1.8 since rounding goes towards even
    // numbers when the decimal part is exactly 0.5 (which it is here).
    Values truthHorizon2(S);
    truthHorizon2 << 0.0, -1.8, -2.0, -2.0,
                    -1.8, -2.0, -2.0, -2.0,
                    -2.0, -2.0, -2.0, -1.8,
                    -2.0, -2.0, -1.8,  0.0;

    ev.setHorizon(2);
    solution = ev(randomPolicy);
    checkSolution(truthHorizon2, std::get<1>(solution));

    Values truthHorizon3(S);
    truthHorizon3 << 0.0, -2.4, -2.9, -3.0,
                    -2.4, -2.9, -3.0, -2.9,
                    -2.9, -3.0, -2.9, -2.4,
                    -3.0, -2.9, -2.4,  0.0;

    ev.setHorizon(3);
    solution = ev(randomPolicy);
    checkSolution(truthHorizon3, std::get<1>(solution));

    Values truthHorizon10(S);
    truthHorizon10<< 0.0, -6.1, -8.4, -9.0,
                    -6.1, -7.7, -8.4, -8.4,
                    -8.4, -8.4, -7.7, -6.1,
                    -9.0, -8.4, -6.1,  0.0;

    ev.setHorizon(10);
    solution = ev(randomPolicy);
    checkSolution(truthHorizon10, std::get<1>(solution));
}
