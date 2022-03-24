#define BOOST_TEST_MODULE Bandit_SuccessiveRejectsPolicy
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include "GlobalFixtures.hpp"

#include <array>
#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Bandit/Experience.hpp>
#include <AIToolbox/Bandit/Model.hpp>
#include <AIToolbox/Bandit/Policies/SuccessiveRejectsPolicy.hpp>

BOOST_AUTO_TEST_CASE( sampling ) {
    using namespace AIToolbox;

    constexpr unsigned budget = 30;

    // Three arms: 0, -1, +1
    Bandit::Model<std::uniform_real_distribution<double>> bandit(
        std::make_tuple(), std::make_tuple(-1.0), std::make_tuple(1.0)
    );

    Bandit::Experience exp(bandit.getA());
    Bandit::SuccessiveRejectsPolicy sr(exp, budget);

    // Fixed constants to check Nk
    const double n = budget;
    const double K = bandit.getA();
    const unsigned nK0 = 0;
    // logBarK for 3 actions
    const double logBarK = 0.5 + 0.5 + 1.0 / 3.0;

    // Check initial phase and Nk
    BOOST_CHECK_EQUAL(sr.canRecommendAction(), false);
    BOOST_CHECK_EQUAL(sr.getCurrentPhase(), 1);

    const unsigned nK1 = std::ceil((1.0 / logBarK) * ((n - K) / (K + 1 - sr.getCurrentPhase())));
    BOOST_CHECK_EQUAL(sr.getCurrentNk(), nK1);

    for (size_t a = 0; a < bandit.getA(); ++a) {
        for (size_t i = 0; i < (nK1 - nK0); ++i) {
            BOOST_CHECK_EQUAL(sr.sampleAction(), a);
            exp.record(a, bandit.sampleR(a));
            sr.stepUpdateQ();
        }
    }

    // Now we should have eliminated hopefully action 1 (with mean -1).

    // Check initial phase and Nk
    BOOST_CHECK_EQUAL(sr.canRecommendAction(), false);
    BOOST_CHECK_EQUAL(sr.getCurrentPhase(), 2);

    const unsigned nK2 = std::ceil((1.0 / logBarK) * ((n - K) / (K + 1 - sr.getCurrentPhase())));
    BOOST_CHECK_EQUAL(sr.getCurrentNk(), nK2);

    // We skip action 1 here since we should have eliminated it.
    //
    // NOTE: This test is not 100% correct; SRP does not necessarily guarantee
    // that we pick the actions in order (i.e. here we could first pull action
    // 2 a bunch then action 0 a bunch). It just so happens that for now this
    // is simpler and it works. However, this behaviour is not guaranteed.
    for (size_t a = 0; a < bandit.getA(); a += 2) {
        for (size_t i = 0; i < (nK2 - nK1); ++i) {
            BOOST_CHECK_EQUAL(sr.sampleAction(), a);
            exp.record(a, bandit.sampleR(a));
            sr.stepUpdateQ();
        }
    }

    // Check we have respected the budget.
    // (This check shouldn't be necessary since we computed the nKs here in the
    // test but it's just a sanity check for the test itself)
    BOOST_CHECK(budget >= (nK1 * 3 + (nK2 - nK1) * 2));

    // Finally check that we can recommend an action, and we recommend the correct one.
    BOOST_CHECK_EQUAL(sr.getCurrentPhase(), 3);
    BOOST_CHECK_EQUAL(sr.canRecommendAction(), true);
    BOOST_CHECK_EQUAL(sr.recommendAction(), 2);
}

