#include <fstream>
#include <iostream>
#include <string>

#include <AIToolbox/Experience.hpp>
#include <AIToolbox/MDP/RLModel.hpp>
#include <AIToolbox/MDP/Utils.hpp>
#include <AIToolbox/Policy.hpp>
#include <AIToolbox/MDP/ValueIteration.hpp>
#include <AIToolbox/MDP/QGreedyPolicy.hpp>
#include <AIToolbox/MDP/PrioritizedSweeping.hpp>

#include <boost/filesystem.hpp>

using std::cout;
using std::cerr;

int main(int argc, char * argv[]) {
    size_t S = 96, A = 2;

    if ( argc < 2 || argc > 3 ) {
        cerr << "Usage: solve_mdp filename [debug]\n";
        return -1;
    }
    AIToolbox::Experience t(S, A);

    bool debug = false;
    if ( argc == 3 ) {
        boost::system::error_code returnedError;
        boost::filesystem::create_directories( "debug", returnedError );
        if ( returnedError ) {
            cerr << "ERR -- Could not create directory 'debug', debug files will not be created.\n";
        }
        else {
            debug = true;
        }
    }

    // LOADING TABLE
    cout << "Loading Table.\n\n";
    {
        std::ifstream tableFile(argv[1]);
        if ( ! ( tableFile >> t ) ) {
            cerr << "ERR -- Could not load specified table.\n";
            return 1;
        }
    }
    cout << "Table loaded correctly.\n\n";

    // OUTPUT LOADED TABLE
    if ( debug ) {
        cout << "DBG -- Outputting table for sanity check...\n";

        std::ofstream tableFile("debug/table_sanity.txt");
        tableFile << t;

        cout << "DBG -- Done.\n\n";
    }

    // NORMALIZING DATA
    cout << "Extracting MDP...\n";
    AIToolbox::MDP::RLModel mdp(t, true);
    cout << "MDP extracted.\n\n";

    if ( debug ) {
        cout << "DBG -- Saving MDP to file...\n";
        {
            std::ofstream outfile("debug/transitionprobabilities_sanity.txt");
            outfile.precision(4);
            outfile << std::fixed;
            int counter = 1;
            for ( size_t a = 0; a < A; a++ ) {
                for ( size_t i = 0; i < S; i++ ) {
                    for ( size_t j = 0; j < S; j++ ) {
                        if ( ! ( counter % 21) ) { outfile << "\t\t\t"; counter = 1; }
                        outfile << mdp.getTransitionFunction()[i][j][a] << "\t";
                        counter ++;
                    }
                    outfile << "\n";
                    counter = 1;
                }
                outfile << "\n\n\n\n\n";
                counter = 1;
            }
        }
        {
            std::ofstream outfile("debug/rewardsnormalized_sanity.txt");
            outfile.precision(4);
            outfile << std::fixed;
            int counter = 1;
            for ( size_t a = 0; a < A; a++ ) {
                for ( size_t i = 0; i < S; i++ ) {
                    for ( size_t j = 0; j < S; j++ ) {
                        if ( ! ( counter % 21) ) { outfile << "\t\t\t"; counter = 1; }
                        outfile << mdp.getRewardFunction()[i][j][a] << "\t";
                        counter ++;
                    }
                    outfile << "\n";
                    counter = 1;
                }
                outfile << "\n\n\n\n\n";
                counter = 1;
            }
        }
        cout << "DBG -- MDP saved.\n\n";
    }

    // SOLVING MDP
    cout << "Making Solver...\n";
    AIToolbox::MDP::ValueIteration solver;
    cout << "Done.\n\n";

    cout << "Solving MDP...\n";
    auto solution = solver(mdp); // Tuple(solved, VFunction, QFunction)
    cout << "MDP Solved.\n";
    cout << "+--> Did we actually solve the MDP? " << ( std::get<0>(solution) ? "YES": "NO" ) << "\n\n";

    {
        std::ofstream outfile("qfun.txt");
        for (size_t s = 0; s < S; s++)
            for (size_t a = 0; a < A; a++)
                outfile << s << " " << a << " " << std::get<2>(solution)[s][a] << "\n";
    }

    // CREATING POLICY
    cout << "Creating QPolicy...\n";
    AIToolbox::MDP::QGreedyPolicy qp( std::get<2>(solution) );

    cout << "Creating Policy...\n";
    AIToolbox::Policy p(qp);
    {
        std::ofstream outfile("policy.txt");
        p.prettyPrint(outfile);
    }
    cout << "Policy created.\n\n";

    // Checking policy with Qtable:
    std::ofstream pcomplete("policy_full.txt");
    pcomplete << p;
    std::ofstream qcomplete("qpolicy_full.txt");
    qcomplete << qp;

    return 0;
}
