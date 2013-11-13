#include "Table.h"

#include <fstream>
#include <array>

#include <iostream>
using std::cout;

Table::Table(size_t Ss, size_t Aa) : S(Ss), A(Aa) {
    table_.resize(S);

    // Clear table
    for ( size_t i = 0; i < S; i++ ) {
        table_[i].resize(S);

        for ( size_t j = 0; j < S; j++ ) {
            table_[i][j].reserve(A);
            for ( size_t a = 0; a < A; a++ ) {
                table_[i][j].emplace_back(std::make_tuple(0, 0.0));
            }
        }
    }

    isValid_ = true;
}

bool Table::load(const std::string & filename) {
    std::ifstream file(filename.c_str(), std::ifstream::in);

    for ( size_t i = 0; i < S; i++ ) {
        for ( size_t j = 0; j < S; j++ ) {
            for ( size_t a = 0; a < A; a++ ) {
                if ( !(file >> std::get<visits>(table_[i][j][a]) >> std::get<reward>(table_[i][j][a]) )) {
             //   if ( !(file >> std::get<visits>(table_[i][j][0]) >> std::get<visits>(table_[i][j][1]) >> std::get<reward>(table_[i][j][0]) >> std::get<reward>(table_[i][j][1]))) {
                    isValid_ = false;
                    return false;
                }
            }
        }
    }
    // Should we verify the data in some way?
    file.close();
    lastFilename_ = filename;
    isValid_ = true;
    return true;
}

bool Table::save(std::string filename) {
    if ( !isValid_ ) return false;

    if ( filename == "" ) {
        if ( lastFilename_ != "" )
            filename = lastFilename_;
        else
            return false;
    }

    std::ofstream file(filename.c_str(), std::ofstream::out);
    for ( size_t i = 0; i < S; i++ ) {
        for ( size_t j = 0; j < S; j++ ) {
            for ( size_t a = 0; a < A; a++ ) {
                file << std::get<visits>(table_[i][j][a]) << " " << std::get<reward>(table_[i][j][a]) << " ";
            }
        }
        file << "\n";
    }

    file.close();
    return true;
}

bool Table::isValid() {
    return isValid_;
}

std::tuple<MDPToolbox::MDP::TransitionTable, MDPToolbox::MDP::RewardTable> Table::getMDP() const {
    MDPToolbox::MDP::TransitionTable P;
    MDPToolbox::MDP::RewardTable R;

    P.resize(S);
    R.resize(S);

    for ( size_t s = 0; s < S; s++ ) {
        P[s].resize(S);
        R[s].resize(S);

        for ( size_t s1 = 0; s1 < S; s1++ ) {
            P[s][s1].resize(A);
            R[s][s1].resize(A);
        }
    }

    double actionSum[S][A];
    for ( size_t s = 0; s < S; s++ )
        for ( size_t a = 0; a < A; a++ )
            actionSum[s][a] = 0.0;

    for ( size_t i = 0; i < S; i++ ) {
        for ( size_t j = 0; j < S; j++ ) {
            for ( size_t a = 0; a < A; a++ ) {
                // Transfering values so that we can manipulate them easier..
                P[i][j][a] = std::get<Table::visits>(table_[i][j][a]);
                R[i][j][a] = std::get<Table::reward>(table_[i][j][a]);
                // actionSum contains the time we have executed action 'a' in state 's'
                actionSum[i][a] += P[i][j][a];
            }
        }
    }
    // Normalize
    for ( size_t i = 0; i < S; i++ ) {
        for ( size_t j = 0; j < S; j++ ) {
            for ( size_t a = 0; a < A; a++ ) {
                // If we never executed 'a' during 'i'
                if ( actionSum[i][a] == 0 ) {
                    // Create shadow state since we never encountered it
                    if ( i == j )
                        P[i][j][a] = 1;
                    else
                        P[i][j][a] = 0;
                    // Reward is already 0 anyway
                }
                else {
                    // Normalize action reward over transition visits
                    if ( P[i][j][a] != 0 ) {
                        R[i][j][a] /= P[i][j][a];
                    }
                    // Normalize transition probability (times we went to 'j' / times we executed 'a' in 'i'
                    P[i][j][a] /= actionSum[i][a];
                }
            }
        }
    }
    return std::make_tuple(P,R);
}
