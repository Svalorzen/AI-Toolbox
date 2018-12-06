#define BOOST_TEST_MODULE Factored_Test
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Factored/Utils/Test.hpp>

#include <iostream>

namespace ai = AIToolbox;
namespace aif = AIToolbox::Factored;

void pprintFV(const aif::FactoredVector & v) {
    for (const auto & e : v) {
        for (const auto i : e.tag)
            std::cout << i << " ";
        std::cout << ": " << e.values.transpose() << '\n';
    }
}

BOOST_AUTO_TEST_CASE( test_1 ) {
    aif::State s{3,3,3};

    aif::FactoredVector A;
    aif::BasisFunction a1{{0,1}, {}};
    a1.values.resize(9);
    a1.values << 1.0, 3.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0;

    aif::BasisFunction a2{{0,2}, {}};
    a2.values.resize(9);
    a2.values << 7.0, 9.0, 8.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0;

    aif::BasisFunction c3{{0,1,2}, {}};
    c3.values.resize(27);
    c3.values << 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                 1., 1., 1., 1., 1., 1., 1., 1., 1.;

    A.emplace_back(std::move(a1));
    A.emplace_back(std::move(a2));

    aif::FactoredVector R;
    aif::BasisFunction r1{{1,2}, {}};
    r1.values.resize(9);
    r1.values << 6.0, 9.0, 5.0, 8.0, 14.0, 1.0, 2.0, 9.0, 11.0;

    aif::BasisFunction r2{{0,2}, {}};
    r2.values.resize(9);
    r2.values << 9.0, 19.0, 12.0, 22.0, 30.0, 27.0, 12.0, 25.0, 1.0;

    R.emplace_back(std::move(r1));
    R.emplace_back(std::move(r2));

    ai::Matrix2D p1(9, 3); // 0,1->0
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
    ai::Matrix2D p2(9, 3);
    p2 <<
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
    ai::Matrix2D p3(9, 3);
    p3 <<
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
    aif::FactoredMatrix f1{{0,1}, p1};
    aif::FactoredMatrix f2{{1,2}, p2};
    aif::FactoredMatrix f3{{0,2}, p3};
    auto T = aif::Factored2DMatrix{f1, f2, f3};

    ai::Vector w(2); w << 2.0, 3.0;
    double discount = 0.5;

     aif::PartialFactorsEnumerator e(s);

    auto Q = aif::bellmanEquation(s, discount, T, A, w, R);

    // pprintFV(Q);

    // while (e.isValid()) {
    //     std::cout << "State: ";
    //     for (auto s : (*e).second)
    //         std::cout << s << " ";
    //     std::cout << "; Q: " << getValue(s, Q, (*e).second);
    //     std::cout << '\n';
    //     e.advance();
    // }
    // std::cout << "####\n";
    // e.reset();

    auto aw = A * w;
    aif::FactoredVector v1bp = aif::backProject(s, T, A * w);
    pprintFV(v1bp);
    std::cout << "^ v1bp ####\n";

    // Q = R + gamma * T * (A * w)
    while (e.isValid()) {
        std::cout << "State: ";
        for (auto s : (*e).second)
            std::cout << s << " ";

        std::cout << '\n';
        double v1 = 0.0, v2 = 0.0, v3 = 0.0;

        aif::PartialFactorsEnumerator s1(s);
        while (s1.isValid()) {
            v1 += getValue(s, T, *e, *s1) *
                  getValue(s, A * w, (*s1).second);

            // std::cout << "Adding " << getValue(s, A * w, (*s1).second) << " * " << getValue(s, T, *e, *s1)
            //           << " == " << getValue(s, T, *e, *s1) * getValue(s, A * w, (*s1).second)
            //           << "\n";

            auto id = aif::toIndexPartial(aw[0].tag, s, *s1);
            v2 += getValue(s, T, *e, *s1) * aw[0].values[id];

            auto id2 = aif::toIndexPartial(aw[1].tag, s, *s1);
            v3 += getValue(s, T, *e, *s1) * aw[1].values[id2];

            std::cout << "Adding " << aw[0].values[id] << " * " << getValue(s, T, *e, *s1)
                      << " == " << getValue(s, T, *e, *s1) * aw[0].values[id]
                      << "        ";

            std::cout << "Adding " << aw[1].values[id2] << " * " << getValue(s, T, *e, *s1)
                      << " == " << getValue(s, T, *e, *s1) * aw[1].values[id2]
                      << "\n";

            s1.advance();
        }

        std::cout << "\n; Qtrue: " <<
            getValue(s, R, (*e).second) + discount * v1
                  << "; Qtruebp: " <<
            getValue(s, R, (*e).second) + discount * getValue(s, v1bp, (*e).second);
        std::cout << "; V1 BP: " << getValue(s, v1bp, (*e).second)
                  << "; V1   : " << v1 << "; V2: " << v2 << "; V3: " << v3;
        std::cout << '\n';
        std::cout << '\n';
        std::cout << '\n';

        BOOST_CHECK(ai::checkEqualGeneral(getValue(s, Q, (*e).second), getValue(s, R, (*e).second) + discount * v1));
        BOOST_CHECK(ai::checkEqualGeneral(getValue(s, Q, (*e).second), getValue(s, R, (*e).second) + discount * getValue(s, v1bp, (*e).second)));
        e.advance();
    }
    std::cout << "####\n";
}

