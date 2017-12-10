#include <AIToolbox/Game/Policies/ESRLPolicy.hpp>

#include <AIToolbox/Utils/Core.hpp>

namespace AIToolbox::Game {
    ESRLPolicy::ESRLPolicy(size_t A, double a, unsigned timesteps, unsigned explorationPhases, unsigned window) :
        Base(A),
        exploit_(false), bestAction_(0),
        timestep_(0), N_(timesteps),
        explorations_(0), explorationPhases_(explorationPhases),
        average_(0.0), window_(window),
        values_(A), allowedActions_(A),
        lri_(A, a)
    {
        values_.fill(0.0);
        std::iota(std::begin(allowedActions_), std::end(allowedActions_), 0);
    }

    void ESRLPolicy::stepUpdateP(size_t a, bool result) {
        if (explorations_ < explorationPhases_) {
            // Check that the action was in our allowed ones.
            const auto it = std::find(std::begin(allowedActions_), std::end(allowedActions_), a);
            if (it == std::end(allowedActions_))
                return;
            // Exploration phase
            lri_.stepUpdateP(std::distance(std::begin(allowedActions_), it), result);

            ++timestep_;
            average_ = ((timestep_ - 1) * average_ + static_cast<double>(result)) / window_;

            // Synchronization phase
            if (N_ >= timestep_) {
                ++explorations_;

                size_t convergedActionLri = 0;
                double maxP = lri_.getActionProbability(0);
                for (size_t i = 1; i < lri_.getA(); ++i) {
                    if (lri_.getActionProbability(i) > maxP) {
                        maxP = lri_.getActionProbability(i);
                        convergedActionLri = i;
                    }
                }

                const auto convergedAction = allowedActions_[convergedActionLri];

                values_[convergedAction] = std::max(values_[convergedAction], average_);

                if (allowedActions_.size() > 1) {
                    std::swap(allowedActions_[convergedActionLri], allowedActions_[allowedActions_.size()-1]);
                    allowedActions_.pop_back();
                } else {
                    allowedActions_.resize(A);
                    std::iota(std::begin(allowedActions_), std::end(allowedActions_), 0);
                }

                lri_ = LRPPolicy(allowedActions_.size(), lri_.getAParam());

                timestep_ = 0;
            }
        } else if (!exploit_) {
            // Exploitation phase
            exploit_ = true;
            values_.maxCoeff(&bestAction_);
        }
    }

    size_t ESRLPolicy::sampleAction() const {
        if (exploit_) return bestAction_;

        return allowedActions_[lri_.sampleAction()];
    }

    double ESRLPolicy::getActionProbability(const size_t & a) const {
        if (exploit_) return a == bestAction_;

        const auto it = std::find(std::begin(allowedActions_), std::end(allowedActions_), a);
        if (it == std::end(allowedActions_))
            return 0.0;

        return lri_.getActionProbability(std::distance(std::begin(allowedActions_), it));
    }

    Vector ESRLPolicy::getPolicy() const {
        Vector retval(A);
        retval.fill(0.0);

        if (exploit_) {
            retval[bestAction_] = 1.0;
            return retval;
        }

        for (size_t i = 0; i < allowedActions_.size(); ++i)
            retval[allowedActions_[i]] = lri_.getActionProbability(i);

        return retval;
    }

    bool ESRLPolicy::isExploiting() const { return exploit_; }
    void ESRLPolicy::setAParam(double a) { lri_.setAParam(a); }
    double ESRLPolicy::getAParam() const { return lri_.getAParam(); }
    void ESRLPolicy::setTimesteps(unsigned t) { N_ = t; }
    unsigned ESRLPolicy::getTimesteps() const { return N_; }
    void ESRLPolicy::setExplorationPhases(unsigned p) { explorationPhases_ = p; }
    unsigned ESRLPolicy::getExplorationPhases() const { return explorationPhases_; }
    void ESRLPolicy::setWindowSize(unsigned window) { window_ = window; }
    unsigned ESRLPolicy::getWindowSize() const { return window_; }
}
