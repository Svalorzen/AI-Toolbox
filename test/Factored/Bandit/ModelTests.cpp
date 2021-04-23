#define BOOST_TEST_MODULE Factored_Bandit_Model

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Factored/Bandit/Model.hpp>

namespace fb = AIToolbox::Factored::Bandit;

BOOST_AUTO_TEST_CASE( construction ) {
    AIToolbox::Factored::Action A{2,2,2};

    std::vector<AIToolbox::Factored::PartialKeys> groups{{0,1},{1,2}};

    AIToolbox::Bandit::Model<std::normal_distribution<double>> testa(std::vector<std::tuple<double, double>>{
            {0.4, 1.0}, // 0-0
            {-.5, 2.0}, // 1-0
            {1.5, 1.0}, // 0-1
            {-.7, 2.0}  // 1-1
    });

    AIToolbox::Bandit::Model<std::normal_distribution<double>> testb(std::vector<std::tuple<double, double>>{
            {-.3, 1.0}, // 0-0
            {-.5, 2.0}, // 1-0
            {0.1, 1.0}, // 0-1
            {-.9, 2.0}  // 1-1
    });

    AIToolbox::Factored::Bandit::Model<std::normal_distribution<double>> bandit(std::move(A), std::move(groups), {std::move(testa), std::move(testb)});
}
