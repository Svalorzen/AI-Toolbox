#define BOOST_TEST_MODULE Factored_BayesianNetwork
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>
#include <AIToolbox/Factored/Utils/BayesianNetwork.hpp>

namespace ai = AIToolbox;
namespace aif = AIToolbox::Factored;

BOOST_AUTO_TEST_CASE( back_projection ) {
    aif::State s{3,3,3};

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
    aif::DBN::Node f1{{0,1}, p};
    aif::DBN::Node f2{{1,2}, p};
    aif::DBN::Node f3{{0,2}, p};
    auto T = aif::DynamicBayesianNetwork{{f1, f2, f3}};

    ai::Vector w(2); w << 2.0, 3.0;

    aif::FactoredVector vbp = aif::backProject(s, T, A * w);

    auto Aw = A * w;

    aif::PartialFactorsEnumerator e(s);
    while (e.isValid()) {
        double valueBP = vbp.getValue(s, (*e).second);
        double value = 0.0;

        aif::PartialFactorsEnumerator s1(s);
        while (s1.isValid()) {
            value +=  T.getTransitionProbability(s, *e, *s1) *
                  Aw.getValue(s, (*s1).second);

            s1.advance();
        }

        BOOST_TEST_INFO("Value: " << value << "; Backprop V: " << valueBP);
        BOOST_CHECK(ai::checkEqualGeneral(value, valueBP));

        e.advance();
    }
}

BOOST_AUTO_TEST_CASE( compact_ddn ) {
    aif::State s{3,3,3};
    ai::Matrix2D p1(9, 3); // x,y->z
    p1 <<
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

    ai::Matrix2D p2(3, 3);
    p2.setIdentity();

    std::vector<aif::DBN::Node> f {
        {{0,1}, p1},
        {{1,2}, p1},
        {{0,2}, p1}
    };

    aif::CompactDDN::Node d1{0, {{0}, p2}};
    aif::CompactDDN::Node d2{1, {{1}, p2}};
    aif::CompactDDN::Node d3{2, {{2}, p2}};

    auto T = aif::CompactDDN({{d1}, {d2}, {d3}}, {f});

    for (size_t a = 0; a < 3; ++a) {
        auto tmp = T.makeDiffTransition(a);

        for (size_t i = 0; i < 3; ++i) {
            if (i == a) {
                BOOST_CHECK_EQUAL(tmp[i].matrix, p2);
                BOOST_CHECK_EQUAL(tmp[i].tag.size(), 1);
                BOOST_CHECK_EQUAL(tmp[i].tag[0], i);
            } else {
                BOOST_CHECK_EQUAL(tmp[i].matrix, p1);
                BOOST_CHECK_EQUAL(ai::veccmp(tmp[i].tag, f[i].tag), 0);
            }
        }
    }
}
