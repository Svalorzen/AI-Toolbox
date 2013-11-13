#include <fstream>
#include <iostream>
#include <string>

#include <MDPToolbox/MDP.hpp>
#include <MDPToolbox/Policy.hpp>
#include <MDPToolbox/Experience.hpp>

using std::cout;
using std::cerr;

int main(int argc, char * argv[]) {
    if ( argc < 2 || argc > 2 ) {
        cerr << "Usage: solve_mdp filename\n";
        return -1;
    }
    MDPToolbox::Experience t(96, 2);

    t.load(argv[1]);

    if ( ! t.isValid() ) {
        cerr << "Could not load specified table.\n";
        return 1;
    }

    //cout << "Table output for sanity check...\n";
    //t.save("table_sanity.txt");

    cout << "Table loaded correctly.\n";
    cout << "Loading table in MDPToolbox...\n";

    MDPToolbox::MDP mdp(96, 2);

    auto mdpdata = t.getMDP();
    cout << "MDP extracted.\n";

    cout << "Saving MDP to file...\n";
    {
        std::ofstream outfile("transitionprobabilities_sanity.txt");
        outfile.precision(4);
        outfile << std::fixed;
        int counter = 1;
        for ( int a = 0; a < 2; a++ ) {
            for ( int i = 0; i < 96; i++ ) {
                for ( int j = 0; j < 96; j++ ) {
                    if ( ! ( counter % 21) ) { outfile << "\t\t\t"; counter = 1; }
                    outfile << std::get<0>(mdpdata)[i][j][a] << "\t";
                    counter ++;
                }
                outfile << "\n";
                counter = 1;
            }
            outfile << "\n\n\n\n\n";
            counter = 1;
        }
        outfile.close();
    }
    {
        std::ofstream outfile("rewardsnormalized_sanity.txt");
        outfile.precision(4);
        outfile << std::fixed;
        int counter = 1;
        for ( int a = 0; a < 2; a++ ) {
            for ( int i = 0; i < 96; i++ ) {
                for ( int j = 0; j < 96; j++ ) {
                    if ( ! ( counter % 21) ) { outfile << "\t\t\t"; counter = 1; }
                    outfile << std::get<1>(mdpdata)[i][j][a] << "\t";
                    counter ++;
                }
                outfile << "\n";
                counter = 1;
            }
            outfile << "\n\n\n\n\n";
            counter = 1;
        }
        outfile.close();
    }
    cout << "MDP saved.\n";

    mdp.setMDP(mdpdata);
    cout << "Table loaded.\n";

    bool out;
    auto p = mdp.valueIteration(&out);

    cout << "Did we actually solve the MDP? " << out << "\n";

    cout << "Policy created.\n";
    {
        std::ofstream outfile("policy.txt");
        outfile << p;
        outfile.close();
    }
    return 0;
}
