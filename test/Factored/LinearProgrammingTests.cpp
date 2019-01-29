#define BOOST_TEST_MODULE Factored_MDP_LinearProgramming
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Factored/MDP/Algorithms/LinearProgramming.hpp>

#include "Utils/SysAdmin.hpp"

#include <iostream>
#include <iomanip>

using std::cout;

std::ostream& operator<<(std::ostream &os, const std::vector<size_t> & v) {
    for (size_t i = 0; i < v.size() - 1; ++i)
        os << v[i] << ' ';
    os << v.back();
    return os;
}

BOOST_AUTO_TEST_CASE( solver ) {
    auto problem = makeSysAdminUniRing(2, 0.1, 0.2, 0.3, 0.4, 0.2, 0.2, 0.1);

    auto rw = problem.getRewardFunction();
    for (auto x : rw.bases) {
        std::cout << "A[" << x.actionTag << "] S[" << x.tag << "]\n";
        for (int i = 0; i < x.values.rows(); ++i) {
            std::cout << "    ";
            for (int y = 0; y < x.values.cols(); ++y)
                cout << std::setw(5) << x.values(i, y);
            cout << '\n';
        }
    }

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
    auto solution = solver(problem, singleBasis, true);
    std::cout << "Done: " << solution.size() << "\n" << solution.transpose() << '\n';
}
