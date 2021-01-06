#define BOOST_TEST_MODULE Factored_MDP_CooperativeQLearning
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Factored/MDP/Algorithms/SparseCooperativeQLearning.hpp>
#include <AIToolbox/Factored/MDP/Algorithms/CooperativeQLearning.hpp>

#include <AIToolbox/Factored/MDP/Environments/SysAdmin.hpp>

namespace aif = AIToolbox::Factored;
namespace fm = AIToolbox::Factored::MDP;

// Our goal here is simply to verify that CooperativeQLearning is doing the
// exact things as SparseCooperativeQLearning.

BOOST_AUTO_TEST_CASE( simple_rule_update ) {
    auto problem = fm::makeSysAdminUniRing(2, 0.1, 0.2, 0.3, 0.4, 0.4, 0.4, 0.3);

    std::vector<std::vector<size_t>> domains{
        {0, 1},
        {2, 3}
    };

    const double alpha = 0.3;

    // Initialize CQL
    fm::CooperativeQLearning cql(problem.getGraph(), domains, problem.getDiscount(), alpha);

    // Initialize SCQL with rules equivalent to the dense CQL QFunction

    std::vector<fm::QFunctionRule> rules;

    const auto & ddngraph = problem.getGraph();
    for (const auto & domain : domains) {
        aif::PartialKeys aTag;
        aif::PartialKeys sTag;
        for (auto d : domain) {
            // Compute state-action domain for this Q factor.
            aTag = aif::merge(aTag, ddngraph.getNodes()[d].agents);
            for (const auto & n : ddngraph.getNodes()[d].parents)
                sTag = aif::merge(sTag, n);
        }

        aif::PartialFactorsEnumerator se(problem.getS(), sTag);
        aif::PartialFactorsEnumerator ae(problem.getA(), aTag);

        while (se.isValid()) {
            ae.reset();
            while (ae.isValid()) {
                rules.emplace_back(*se, *ae, 0.0);
                ae.advance();
            }
            se.advance();
        }
    }

    fm::SparseCooperativeQLearning scql(problem.getS(), problem.getA(), problem.getDiscount(), alpha);
    for (const auto & rule : rules)
        scql.insertRule(rule);

    // Run some random experiences and verify that the computed optimal actions
    // are the same.
    aif::State   s(problem.getS().size());
    aif::State  s1(problem.getS().size());
    aif::Action  a(problem.getA().size());
    aif::Rewards r(problem.getA().size());

    AIToolbox::RandomEngine rnd;

    std::vector<std::uniform_int_distribution<size_t>> Sdists;
    for (auto s : problem.getS()) Sdists.emplace_back(0, s-1);

    std::vector<std::uniform_int_distribution<size_t>> Adists;
    for (auto a : problem.getA()) Adists.emplace_back(0, a-1);

    std::uniform_real_distribution<double> Rdist(0, 10);

    for (size_t i = 0; i < 100; ++i) {
        // Generate S and S'
        for (size_t j = 0; j < s.size(); ++j) {
            s[j]  = Sdists[j](rnd);
            s1[j] = Sdists[j](rnd);
        }
        // Generate A and R
        for (size_t j = 0; j < a.size(); ++j) {
            a[j] = Adists[j](rnd);
            r[j] = Rdist(rnd);
        }

        const auto cqla1  = cql.stepUpdateQ(s, a, s1, r);
        const auto scqla1 = scql.stepUpdateQ(s, a, s1, r);

        // Check actions are equal
        BOOST_CHECK_EQUAL(AIToolbox::veccmp(cqla1, scqla1), 0);
    }

    // Finally, check that the QFunction is the same for both methods.

    const auto & qf = cql.getQFunction();
    const auto & qr = scql.getQFunctionRules();

    aif::PartialFactorsEnumerator sDomain(problem.getS());
    aif::PartialFactorsEnumerator aDomain(problem.getA());

    for (size_t sId = 0; sDomain.isValid(); sDomain.advance(), ++sId) {
        for (size_t aId = 0; aDomain.isValid(); aDomain.advance(), ++aId) {
            double vcql = 0.0, vscql = 0.0;
            for (const auto & qfbasis : qf.bases) {
                const auto sip = aif::toIndexPartial(qfbasis.tag, problem.getS(), *sDomain);
                const auto aip = aif::toIndexPartial(qfbasis.actionTag, problem.getA(), *aDomain);

                vcql += qfbasis.values(sip, aip);
            }

            auto rules = qr.filter(aif::join(sDomain->second, aDomain->second));
            // Make sure that the number of applicable rules is the same as the
            // number of basis functions (sanity check).
            BOOST_CHECK_EQUAL(rules.size(), qf.bases.size());
            for (const auto & rule : rules)
                vscql += rule.value;

            BOOST_CHECK_EQUAL(vcql, vscql);
        }
        aDomain.reset();
    }
}
