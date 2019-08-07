#include <AIToolbox/Factored/Bandit/Algorithms/MAUCE.hpp>

#include <AIToolbox/Factored/Utils/Core.hpp>

#include <AIToolbox/Impl/Logging.hpp>

namespace AIToolbox::Factored::Bandit {
    MAUCE::MAUCE(Action aa, const std::vector<PartialKeys> & dependencies, std::vector<double> ranges) :
            A(std::move(aa)), timestep_(0),
            averages_(A, dependencies), rangesSquared_(std::move(ranges)),
            logA_(0.0)
    {
        assert(dependencies.size() == rangesSquared_.size());
        // Compute log(|A|) without needing to compute |A| which may be too
        // big. We'll use it later to obtain log(t |A|)
        for (const auto a : A)
            logA_ += std::log(a);

        // Square all ranges since that's the only form in which we use them.
        for (auto & r : rangesSquared_)
            r = r * r;
    }

    Action MAUCE::stepUpdateQ(const Action & a, const Rewards & rew) {
        AI_LOGGER(AI_SEVERITY_INFO, "Updating averages...");

        averages_.stepUpdateQ(a, rew);

        // Build the vectors to pass to UCVE
        AI_LOGGER(AI_SEVERITY_INFO, "Populating graph...");

        UCVE::GVE::Graph graph(A.size());

        const auto & q = averages_.getQFunction();
        const auto & c = averages_.getCounts();

        for (size_t x = 0; x < q.bases.size(); ++x) {
            const auto & basis = q.bases[x];
            const auto & cc = c[x];
            auto & factorNode = graph.getFactor(basis.tag)->getData();
            const bool isFilled = factorNode.size() > 0;

            if (!isFilled) factorNode.reserve(basis.values.size());

            for (size_t y = 0; y < static_cast<size_t>(basis.values.size()); ++y) {
                // We give rules we haven't seen yet a headstart so they'll get picked first
                // We divide by the number of groups_ here with the hope that the
                // value itself is still high enough that it shadows the rest of
                // the rules, but it also allows to sum and compare them so that we
                // still get to optimize multiple actions at once (the max would
                // just cap to inf).
                UCVE::V val;
                if (cc[y] == 0) {
                    // If we have already filled this, then this transition has
                    // already been set as seen (since the tag is the same), so
                    // we don't really need to do anything and we skip.
                    if (isFilled) continue;
                    // Otherwise we put the max value to ensure it gets picked.
                    val = {std::numeric_limits<double>::max() / q.bases.size(), 0.0};
                } else {
                    // Otherwise we get the correct value and exploration bonus
                    // from the RollingAverage.
                    val = {basis.values(y), rangesSquared_[x] / cc[y]}; // Range^2 / Count
                }

                if (isFilled) {
                    // We accumulate all values, given that we have seen this at least once.
                    factorNode[y].second[0].v += val;
                } else {
                    factorNode.emplace_back(y, UCVE::Factor{{val, {}}});
                }
            }
        }

        // Update the timestep, and finish computing log(t |A|) for this
        // timestep.
        ++timestep_;
        const auto logtA = logA_ + std::log(timestep_);

        // Create and run UCVE
        AI_LOGGER(AI_SEVERITY_INFO, "Now running UCVE...");
        UCVE ucve;
        auto a_v = ucve(A, logtA, graph);
        AI_LOGGER(AI_SEVERITY_INFO, "Done.");

        return std::get<0>(a_v);
    }

    const RollingAverage & MAUCE::getRollingAverage() const {
        return averages_;
    }

    unsigned MAUCE::getTimestep() const { return timestep_; }
    void MAUCE::setTimestep(unsigned t) { timestep_ = t; }
}
