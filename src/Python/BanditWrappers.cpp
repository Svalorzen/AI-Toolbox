#include <boost/python.hpp>

void exportBanditPolicyInterface();

void exportBanditGreedyPolicy();
void exportBanditThompsonSamplingPolicy();

BOOST_PYTHON_MODULE(Bandit)
{
    boost::python::docstring_options localDocstringOptions(true, true, false);

    exportBanditPolicyInterface();

    exportBanditGreedyPolicy();
    exportBanditThompsonSamplingPolicy();
}
