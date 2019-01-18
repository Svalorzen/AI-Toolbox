#include <AIToolbox/Bandit/Policies/ESRLPolicy.hpp>

#include <AIToolbox/Utils/Core.hpp>

namespace AIToolbox::Bandit {
    ESRLPolicy::ESRLPolicy(size_t A, double a, unsigned timesteps, unsigned explorationPhases, unsigned window) :
        Base(A),
        exploit_(false), bestAction_(0),
        timestep_(0), N_(timesteps),
        explorations_(0), explorationPhases_(explorationPhases),
        average_(0.0), window_(window),
        values_(A), allowedActions_(A),
        lri_(A, a)
    {
        values_.setZero();
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
            // Note that the paper contains an error here. It says that you
            // should multiply average by (timestep - 1)/WINDOW, but that would
            // make no sense.
            average_ = ((window_ - 1) * average_ + static_cast<double>(result)) / window_;

            // Synchronization phase
            if (timestep_ >= N_) {
                ++explorations_;

                // So here we assume that we have converged to something.
                // We extract the action we have converged to as the one that
                // has the highest likelihood of being chosen.
                size_t convergedActionLri = 0;
                double maxP = lri_.getActionProbability(0);
                for (size_t i = 1; i < lri_.getA(); ++i) {
                    if (lri_.getActionProbability(i) > maxP) {
                        maxP = lri_.getActionProbability(i);
                        convergedActionLri = i;
                    }
                }

                // We convert that action in "LRI" space, which may have less
                // action than ours since actions can be banned, to our space.
                const auto convergedAction = allowedActions_[convergedActionLri];

                // The value for the action is updated if it has increased,
                // since the Nash equilibrium we have converged to may be
                // different than the old one if another agent changed its
                // action.
                values_[convergedAction] = std::max(values_[convergedAction], average_);

                // If we have more than one allowed action, we remove it.
                // Otherwise, we reset to allowing the whole spectrum.
                if (allowedActions_.size() > 1) {
                    std::swap(allowedActions_[convergedActionLri], allowedActions_[allowedActions_.size()-1]);
                    allowedActions_.pop_back();
                } else {
                    allowedActions_.resize(A);
                    std::iota(std::begin(allowedActions_), std::end(allowedActions_), 0);
                }

                lri_ = LRPPolicy(allowedActions_.size(), lri_.getAParam());

                timestep_ = 0;
                average_ = 0.0;
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
        retval.setZero();

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
