#include <AIToolbox/PolicyInterface.hpp>
#include <chrono>

namespace AIToolbox {
    PolicyInterface::PolicyInterface(size_t s, size_t a) : S(s), A(a),
                                                           rand_(std::chrono::system_clock::now().time_since_epoch().count()),
                                                           sampleDistribution_(0.0, 1.0) {}

    size_t PolicyInterface::getS() const { return S; }
    size_t PolicyInterface::getA() const { return A; }
}
