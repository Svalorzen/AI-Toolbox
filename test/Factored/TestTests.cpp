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
    aif::State s{2,2,2};

    aif::FactoredVector C;
    aif::BasisFunction c1{{0,1}, {}};
    c1.values.resize(4);
    c1.values << 1.0, 3.0, 2.0, 4.0;

    aif::BasisFunction c2{{0,2}, {}};
    c2.values.resize(4);
    c2.values << 7.0, 9.0, 8.0, 10.0;

    aif::BasisFunction c3{{0,1,2}, {}};
    c3.values.resize(8);
    c3.values << 1., 1., 1., 1., 1., 1., 1., 1.;

    C.emplace_back(std::move(c1));
    C.emplace_back(std::move(c2));
    // C.emplace_back(std::move(c3));

    aif::FactoredVector b;
    aif::BasisFunction b1{{1,2}, {}};
    b1.values.resize(4);
    b1.values << 6.0, 9.0, 5.0, 8.0;

    aif::BasisFunction b2{{0,2}, {}};
    b2.values.resize(4);
    b2.values << 9.0, 19.0, 12.0, 22.0;

    b.emplace_back(std::move(b1));
    b.emplace_back(std::move(b2));

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
    auto P = aif::Factored2DMatrix{f1, f2, f3};

    ai::Vector w(2); w << 2.0, 3.0;

    auto result = aif::bellmanEquation(s, 0.5, P, C, w, b);

    pprintFV(result);

    aif::PartialFactorsEnumerator e(s);
    while (e.isValid()) {
        std::cout << "State: ";
        for (auto s : (*e).second)
            std::cout << s << " ";
        std::cout << "; C: " << getValue(s, result, (*e).second);
        std::cout << '\n';
        e.advance();
    }
    std::cout << "####\n";
}

