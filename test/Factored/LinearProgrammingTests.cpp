#define BOOST_TEST_MODULE Factored_MDP_LinearProgramming
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Factored/MDP/Algorithms/LinearProgramming.hpp>

#include "Utils/SysAdmin.hpp"

#include <iostream>

BOOST_AUTO_TEST_CASE( solver ) {
    auto problem = makeSysAdminUniRing(3, 0.1, 0.2, 0.3, 0.4, 0.2, 0.2, 0.1);

    auto singleBasis = aif::FactoredVector();
    for (size_t s = 0; s < problem.getS().size(); ++s) {
        // For each element in the state space, we create 3 BasisFunctions;
        // each indicating one of the three different values.
        singleBasis.bases.emplace_back(aif::BasisFunction{{s}, ai::Vector(3)});
        singleBasis.bases.back().values << 1.0, 0.0, 0.0;

        singleBasis.bases.emplace_back(aif::BasisFunction{{s}, ai::Vector(3)});
        singleBasis.bases.back().values << 0.0, 1.0, 0.0;

        singleBasis.bases.emplace_back(aif::BasisFunction{{s}, ai::Vector(3)});
        singleBasis.bases.back().values << 0.0, 0.0, 1.0;
    }
    std::cout << "Added " << singleBasis.bases.size() << " bases.\n";

    auto solver = afm::LinearProgramming();
    std::cout << solver(problem, singleBasis, true);
}
