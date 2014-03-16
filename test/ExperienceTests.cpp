#define BOOST_TEST_MODULE Experience
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Experience.hpp>

#include <array>
#include <algorithm>
#include <fstream>
#include <cstdio>

BOOST_AUTO_TEST_CASE( construction ) {
    const int S = 5, A = 6;

    AIToolbox::Experience exp(S, A);

    BOOST_CHECK_EQUAL(exp.getS(), S);
    BOOST_CHECK_EQUAL(exp.getA(), A);

    BOOST_CHECK_EQUAL(exp.getVisits(0,0,0), 0);
    BOOST_CHECK_EQUAL(exp.getReward(0,0,0), 0.0);

    BOOST_CHECK_EQUAL(exp.getVisits(S-1,S-1,A-1), 0);
    BOOST_CHECK_EQUAL(exp.getReward(S-1,S-1,A-1), 0.0);
}

BOOST_AUTO_TEST_CASE( recording ) {
    const int S = 5, A = 6;

    AIToolbox::Experience exp(S, A);

    const int s = 3, s1 = 4, a = 5;
    const double rew = 7.4, negrew = -4.2, zerorew = 0.0;

    BOOST_CHECK_EQUAL(exp.getVisits(s,s1,a), 0);

    exp.record(s,s1,a,rew);

    BOOST_CHECK_EQUAL(exp.getVisits(s,s1,a), 1);
    BOOST_CHECK_EQUAL(exp.getReward(s,s1,a), rew);

    exp.reset();

    BOOST_CHECK_EQUAL(exp.getVisits(s,s1,a), 0);

    exp.record(s,s1,a,negrew);

    BOOST_CHECK_EQUAL(exp.getVisits(s,s1,a), 1);
    BOOST_CHECK_EQUAL(exp.getReward(s,s1,a), negrew);

    exp.record(s,s1,a,zerorew);

    BOOST_CHECK_EQUAL(exp.getVisits(s,s1,a), 2);
    BOOST_CHECK_EQUAL(exp.getReward(s,s1,a), negrew);

    BOOST_CHECK_EQUAL(exp.getVisitsSum(s, a), 2);
}

int generator() {
    static int counter = 0;
    return ++counter;
}

BOOST_AUTO_TEST_CASE( compatibility ) {
    const int S = 4, A = 3;
    AIToolbox::Experience exp(S,A);

    std::array<std::array<std::array<int, A>, S>, S> visits;
    std::array<std::array<std::array<int, A>, S>, S> rewards;
    for ( size_t s = 0; s < S; ++s )
        for ( size_t s1 = 0; s1 < S; ++s1 ) {
            std::generate(visits[s][s1].begin(), visits[s][s1].end(), generator);
            std::generate(rewards[s][s1].begin(), rewards[s][s1].end(), generator);
        }

    exp.setVisits(visits);
    exp.setRewards(rewards);

    for ( size_t s = 0; s < S; ++s ) {
        std::vector<int> visitsSum(A,0), rewardSum(A,0);
        for ( size_t s1 = 0; s1 < S; ++s1 ) {
            for ( size_t a = 0; a < A; ++a ) {
                BOOST_CHECK_EQUAL( exp.getVisits(s,s1,a), visits[s][s1][a] );
                BOOST_CHECK_EQUAL( exp.getReward(s,s1,a), rewards[s][s1][a] );
                visitsSum[a] += visits[s][s1][a];
                rewardSum[a] += rewards[s][s1][a];
            }
        }
        for ( size_t a = 0; a < A; ++a ) {
            BOOST_CHECK_EQUAL( exp.getVisitsSum(s,a), visitsSum[a] );
            BOOST_CHECK_EQUAL( exp.getRewardSum(s,a), rewardSum[a] );
        }
    }
}

BOOST_AUTO_TEST_CASE( files ) {
    const int S = 96, A = 2;
    AIToolbox::Experience exp(S,A);

    std::string inputFilename  = "./data/experience.txt";
    std::string outputFilename = "./loadedExperience.txt";
    {
        std::ifstream inputFile(inputFilename);

        if ( !inputFile ) BOOST_FAIL("Data to perform test could not be loaded: " + inputFilename);
        BOOST_CHECK( inputFile >> exp );
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
