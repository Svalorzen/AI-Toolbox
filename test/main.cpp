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
    size_t S = 5, A = S;
    
    Experience exp(S,A);
    MDP::RLModel model(exp, false);
    MDP::PrioritizedSweeping<MDP::RLModel> ps(model, 0.9, 0.01, 200);

    std::default_random_engine rand(0);
    std::uniform_int_distribution<int> dist(0,A-1);
//    std::normal_distribution<double> rew1(4, 2);
//   std::normal_distribution<double> rew2(-3, 1);

    for ( int i = 0; i < 5000; ++i ) {
        size_t s = dist(rand), s1 = dist(rand), a = dist(rand);
        s1 = a;
        double rew;
        if ( s == s1 ) rew = 0;
        else if ( s ) rew = -10;
        else rew = 12;

        exp.record(s,s1,a,rew);
        model.sync(s,a);

        ps.stepUpdateQ(s,a);
        ps.batchUpdateQ();
    }

    AIToolbox::MDP::ValueIteration solver;
    auto solution = solver(model); // Tuple(solved, VFunction, QFunction)
    cout << "MDP Solved.\n";
    cout << "+--> Did we actually solve the MDP? " << ( std::get<0>(solution) ? "YES": "NO" ) << "\n\n";

    // CREATING POLICY
    cout << "Creating QPolicies...\n";
    AIToolbox::MDP::QGreedyPolicy qp1( std::get<2>(solution) );
    auto & q = ps.getQFunction();
    AIToolbox::MDP::QGreedyPolicy qp2( ps.getQFunction() );

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
