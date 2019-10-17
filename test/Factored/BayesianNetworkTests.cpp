#define BOOST_TEST_MODULE Factored_BayesianNetwork
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>
#include <AIToolbox/Factored/Utils/BayesianNetwork.hpp>

namespace ai = AIToolbox;
namespace aif = AIToolbox::Factored;

BOOST_AUTO_TEST_CASE( back_project ) {
    aif::State s{3,3,3};
    aif::Action a{2,2};

    aif::FactoredVector A;
    aif::BasisFunction a1{{0,1}, {}};
    a1.values.resize(9);
    a1.values << 1.0, 3.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0;

    aif::BasisFunction a2{{0,2}, {}};
    a2.values.resize(9);
    a2.values << 7.0, 9.0, 8.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0;

    A.bases.emplace_back(std::move(a1));
    A.bases.emplace_back(std::move(a2));

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
    aif::FactoredDDN::Node f1{{0}, {
        {{0,1}, p}, {{0,2}, p}
    }};
    aif::FactoredDDN::Node f2{{1}, {
        {{0,1}, p}, {{0,2}, p}
    }};
    aif::FactoredDDN::Node f3{{1}, {
        {{0,1}, p}, {{0,2}, p}
    }};
    auto T = aif::FactoredDDN{{f1, f2, f3}};

    ai::Vector w(2); w << 2.0, 3.0;

    auto Aw = A * w;
    aif::FactoredMatrix2D vbp = aif::backProject(s, a, T, Aw);

    aif::PartialFactorsEnumerator es(s);
    aif::PartialFactorsEnumerator ea(a);
    while (es.isValid()) {
        while (ea.isValid()) {
            double valueBP = vbp.getValue(s, a, (*es).second, (*ea).second);
            double value = 0.0;

            aif::PartialFactorsEnumerator s1(s);
            while (s1.isValid()) {
                value += T.getTransitionProbability(s, a, *es, *ea, *s1) *
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
