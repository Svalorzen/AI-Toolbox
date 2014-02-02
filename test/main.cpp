#include <fstream>
#include <iostream>
#include <string>
#include <random>

#include <AIToolbox/Experience.hpp>
#include <AIToolbox/MDP/RLModel.hpp>
#include <AIToolbox/MDP/IO.hpp>
#include <AIToolbox/Policy.hpp>
#include <AIToolbox/MDP/ValueIteration.hpp>
#include <AIToolbox/MDP/QGreedyPolicy.hpp>
#include <AIToolbox/MDP/PrioritizedSweeping.hpp>
#include <AIToolbox/MDP/Utils.hpp>

using std::cout;
using std::cerr;

int main(/*int argc, char * argv[]*/) {
    using namespace AIToolbox;
    size_t S = 3, A = 3;
    
    Experience exp(S,A);
    MDP::RLModel model(exp, false);
    MDP::QFunction q = MDP::makeQFunction(S,A);
    MDP::PrioritizedSweeping ps(S, A, 1, 0.9, 0.01, 200);

    std::default_random_engine rand(0);
    std::uniform_int_distribution<int> dist(0,A-1);

    for ( int i = 0; i < 500; ++i ) {
        size_t s = dist(rand), s1 = dist(rand), a = dist(rand);
        double rew = !( a % 2 ) + 5;

        exp.record(s,s1,a,rew);
        model.sync(s,a);

        ps.stepUpdateQ(s,s1,a,rew, q);
        ps.batchUpdateQ(model, &q);
    }

    AIToolbox::MDP::ValueIteration solver;
    auto solution = solver(model); // Tuple(solved, VFunction, QFunction)
    cout << "MDP Solved.\n";
    cout << "+--> Did we actually solve the MDP? " << ( std::get<0>(solution) ? "YES": "NO" ) << "\n\n";

    // CREATING POLICY
    cout << "Creating QPolicies...\n";
    AIToolbox::MDP::QGreedyPolicy qp1( std::get<2>(solution) );
    AIToolbox::MDP::QGreedyPolicy qp2( q );

    cout << exp << "\n\n";
    cout << model << "\n\n";

    for ( size_t s = 0; s < S; ++s ) {
        for ( size_t a = 0; a < A; ++a ) {
            cout << std::get<2>(solution)[s][a] << "\t";
        }
        cout << "\n";
    }
    cout << "\n\n";

    for ( size_t s = 0; s < S; ++s ) {
        for ( size_t a = 0; a < A; ++a ) {
            cout << q[s][a] << "\t";
        }
        cout << "\n";
    }
    cout << "\n\n";

    for ( size_t s = 0; s < S; ++s ) {
        for ( size_t a = 0; a < A; ++a ) {
            double r = 0;
            for ( size_t s1 = 0; s1 < S; ++s1 ) {
                r += exp.getReward(s, s1, a);
            }
            cout << s << '\t' << a << '\t' << r << '\n';
        }
    }
    cout << '\n';
    cout << qp1 << "\n\n";
    cout << qp2 << "\n\n";

    return 0;
}
