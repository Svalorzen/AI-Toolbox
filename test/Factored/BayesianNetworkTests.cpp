#define BOOST_TEST_MODULE Factored_BayesianNetwork
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include "GlobalFixtures.hpp"

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>
#include <AIToolbox/Factored/Utils/BayesianNetwork.hpp>

namespace ai = AIToolbox;
namespace aif = AIToolbox::Factored;

BOOST_AUTO_TEST_CASE( back_project ) {
    aif::State s{3,3,3};
    aif::Action a{2,2};

    aif::DDNGraph graph(s, a);
    graph.push({{0},
        {{0,1},{0,2}}
    });
    graph.push({{1},
        {{0,1},{0,2}}
    });
    graph.push({{1},
        {{0,1},{0,2}}
    });

    ai::Matrix2D p(9, 3); // x,y->z
    p <<
       0.90, 0.05, 0.05, // 0, 0
       0.70, 0.20, 0.10, // 0, 1
       0.20, 0.50, 0.30, // 0, 2
       0.05, 0.90, 0.05, // 1, 0
       0.10, 0.70, 0.20, // 1, 1
       0.20, 0.50, 0.30, // 1, 2
       0.05, 0.05, 0.90, // 2, 0
       0.20, 0.10, 0.70, // 2, 1
       0.50, 0.10, 0.40  // 2, 2
    ;

    auto T = aif::DDN{graph, {}};
    T.transitions.resize(3);

    // We simply paste p two times in each transition matrix to store the
    // probabilities for every parent set.
    T.transitions[0] = p.replicate<2, 1>();
    T.transitions[1] = p.replicate<2, 1>();
    T.transitions[2] = p.replicate<2, 1>();

    // Setup the matrices for backpropagation.
    aif::FactoredVector A;
    aif::BasisFunction a1{{0,1}, {}};
    a1.values.resize(9);
    a1.values << 1.0, 3.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0;

    aif::BasisFunction a2{{0,2}, {}};
    a2.values.resize(9);
    a2.values << 7.0, 9.0, 8.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0;

    A.bases.emplace_back(std::move(a1));
    A.bases.emplace_back(std::move(a2));

    ai::Vector w(2); w << 2.0, 3.0;

    auto Aw = A * w;
    aif::FactoredMatrix2D vbp = aif::backProject(T, Aw);

    aif::PartialFactorsEnumerator es(s);
    aif::PartialFactorsEnumerator ea(a);
    while (es.isValid()) {
        while (ea.isValid()) {
            double valueBP = vbp.getValue(s, a, (*es).second, (*ea).second);
            double value = 0.0;

            aif::PartialFactorsEnumerator s1(s);
            while (s1.isValid()) {
                value += T.getTransitionProbability(*es, *ea, *s1) *
                      Aw.getValue(s, (*s1).second);

                s1.advance();
            }

            BOOST_TEST_INFO("Value: " << value << "; Backprop V: " << valueBP);
            BOOST_CHECK(ai::checkEqualGeneral(value, valueBP));

            ea.advance();
        }
        ea.reset();
        es.advance();
    }
}
