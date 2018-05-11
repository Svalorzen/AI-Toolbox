#include <boost/python.hpp>

class NamespaceBandit{};

void exportBanditPolicyInterface();

void exportBanditGreedyPolicy();
void exportBanditThompsonSamplingPolicy();

void exportBandit() {
#ifdef AITOOLBOX_EXPORT_BANDIT
    boost::python::scope x = boost::python::class_<NamespaceBandit>("Bandit");

    exportBanditPolicyInterface();

    exportBanditGreedyPolicy();
    exportBanditThompsonSamplingPolicy();
#endif
}
