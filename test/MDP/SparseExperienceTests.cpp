#define BOOST_TEST_MODULE MDP_SparseExperience
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/MDP/SparseExperience.hpp>
#include <AIToolbox/MDP/IO.hpp>

#include <array>
#include <algorithm>
#include <fstream>
#include <cstdio>

BOOST_AUTO_TEST_CASE( construction ) {
    const size_t S = 5, A = 6;

    AIToolbox::MDP::SparseExperience exp(S, A);

    BOOST_CHECK_EQUAL(exp.getS(), S);
    BOOST_CHECK_EQUAL(exp.getA(), A);

    BOOST_CHECK_EQUAL(exp.getVisits(0,0,0), 0);
    BOOST_CHECK_EQUAL(exp.getReward(0,0,0), 0.0);

    BOOST_CHECK_EQUAL(exp.getVisits(S-1,A-1,S-1), 0);
    BOOST_CHECK_EQUAL(exp.getReward(S-1,A-1,S-1), 0.0);
}

BOOST_AUTO_TEST_CASE( recording ) {
    const size_t S = 5, A = 6;

    AIToolbox::MDP::SparseExperience exp(S, A);

    const size_t s = 3, s1 = 4, a = 5;
    const double rew = 7.4, negrew = -4.2, zerorew = 0.0;

    BOOST_CHECK_EQUAL(exp.getVisits(s,a,s1), 0);

    exp.record(s,a,s1,rew);

    BOOST_CHECK_EQUAL(exp.getVisits(s,a,s1), 1);
    BOOST_CHECK_EQUAL(exp.getReward(s,a,s1), rew);

    exp.reset();

    BOOST_CHECK_EQUAL(exp.getVisits(s,a,s1), 0);

    exp.record(s,a,s1,negrew);

    BOOST_CHECK_EQUAL(exp.getVisits(s,a,s1), 1);
    BOOST_CHECK_EQUAL(exp.getReward(s,a,s1), negrew);

    exp.record(s,a,s1,zerorew);

    BOOST_CHECK_EQUAL(exp.getVisits(s,a,s1), 2);
    BOOST_CHECK_EQUAL(exp.getReward(s,a,s1), negrew);

    BOOST_CHECK_EQUAL(exp.getVisitsSum(s, a), 2);
}

int generator() {
    static int counter = 0;
    return ++counter;
}

BOOST_AUTO_TEST_CASE( compatibility ) {
    const size_t S = 4, A = 3;
    AIToolbox::MDP::SparseExperience exp(S,A);

    std::array<std::array<std::array<int, S>, A>, S> visits;
    std::array<std::array<std::array<int, S>, A>, S> rewards;
    for ( size_t s = 0; s < S; ++s )
        for ( size_t a = 0; a < A; ++a ) {
            std::generate(visits[s][a].begin(), visits[s][a].end(), generator);
            std::generate(rewards[s][a].begin(), rewards[s][a].end(), generator);
        }

    exp.setVisits(visits);
    exp.setRewards(rewards);

    for ( size_t a = 0; a < A; ++a ) {
        for ( size_t s = 0; s < S; ++s ) {
            int visitsSum = 0, rewardSum = 0;
            for ( size_t s1 = 0; s1 < S; ++s1 ) {
                BOOST_CHECK_EQUAL( exp.getVisits(s,a,s1), visits[s][a][s1] );
                BOOST_CHECK_EQUAL( exp.getReward(s,a,s1), rewards[s][a][s1] );
                visitsSum += visits[s][a][s1];
                rewardSum += rewards[s][a][s1];
            }
            BOOST_CHECK_EQUAL( exp.getVisitsSum(s,a), visitsSum );
            BOOST_CHECK_EQUAL( exp.getRewardSum(s,a), rewardSum );
        }
    }
}

BOOST_AUTO_TEST_CASE( files ) {
    const size_t S = 96, A = 2;
    AIToolbox::MDP::SparseExperience exp(S,A);

    std::string inputFilename  = "./data/experience.txt";
    std::string outputFilename = "./loadedExperience.txt";
    {
        std::ifstream inputFile(inputFilename);

        if ( !inputFile ) BOOST_FAIL("Data to perform test could not be loaded: " + inputFilename);
        BOOST_CHECK( inputFile >> exp );

        for ( size_t a = 0; a < A; ++a ) {
            for ( size_t s = 0; s < S; ++s ) {
                int visitsSum = 0, rewardSum = 0;
                for ( size_t s1 = 0; s1 < S; ++s1 ) {
                    visitsSum += exp.getVisits(s, a, s1);
                    rewardSum += exp.getReward(s, a, s1);
                }
                BOOST_CHECK_EQUAL( exp.getVisitsSum(s,a), visitsSum );
                BOOST_CHECK_EQUAL( exp.getRewardSum(s,a), rewardSum );
            }
        }
    }
    {
        std::ofstream outputFile(outputFilename);
        if ( !outputFile ) BOOST_FAIL("Could not open file for writing: " + outputFilename);
        BOOST_CHECK( outputFile << exp );
    }
    {
        std::ifstream inputFile(inputFilename);
        std::ifstream writtenFile(outputFilename);

        double input, written;
        while ( inputFile >> input ) {
            BOOST_CHECK( writtenFile >> written );
            BOOST_CHECK_EQUAL( written, input );
        }
        BOOST_CHECK( ! ( writtenFile >> written ) );
    }
    // Cleanup
    {
        std::remove(outputFilename.c_str());
    }
}
