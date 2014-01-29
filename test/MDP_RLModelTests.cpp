#define BOOST_TEST_MODULE MDP_RLModel
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Experience.hpp>
#include <AIToolbox/MDP/RLModel.hpp>

#include <fstream>

BOOST_AUTO_TEST_CASE( construction ) {
    const int S = 10, A = 8;

    AIToolbox::Experience exp(S,A);
    {
        std::ifstream inputExperience("./data/experience.txt"); 
        
        if ( !inputExperience ) BOOST_FAIL("Data to perform test could not be loaded.");
        BOOST_CHECK( inputExperience >> exp );
    }

    AIToolbox::MDP::RLModel model(exp, true);

}
