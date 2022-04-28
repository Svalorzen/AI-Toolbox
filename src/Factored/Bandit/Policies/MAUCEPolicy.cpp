#include <AIToolbox/Factored/Bandit/Policies/MAUCEPolicy.hpp>

#include <AIToolbox/Factored/Utils/Core.hpp>

#include <AIToolbox/Logging.hpp>

namespace AIToolbox::Factored::Bandit {
    MAUCEPolicy::MAUCEPolicy(const Experience & exp, std::vector<double> ranges) :
            Base(exp.getA()), exp_(exp), rangesSquared_(std::move(ranges)),
            logA_(0.0)
    {
        assert(exp_.getRewardMatrix().bases.size() == rangesSquared_.size());
        // Compute log(|A|) without needing to compute |A| which may be too
        // big. We'll use it later to obtain log(t |A|)
        for (const auto a : exp_.getA())
            logA_ += std::log(a);

        // Square all ranges since that's the only form in which we use them.
        for (auto & r : rangesSquared_)
            r = r * r;
    }

    Action MAUCEPolicy::sampleAction() const {
        // Build the vectors to pass to UCVE
        AI_LOGGER(AI_SEVERITY_INFO, "Populating graph...");

        UCVE::GVE::Graph graph(exp_.getA().size());

        const auto & q = exp_.getRewardMatrix();
        const auto & c = exp_.getVisitsTable();

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

        // Finish computing log(t |A|) for this timestep.
        const auto logtA = logA_ + std::log(exp_.getTimesteps());

        // Create and run UCVE
        AI_LOGGER(AI_SEVERITY_INFO, "Now running UCVE...");
        UCVE ucve;
        auto a_v = ucve(exp_.getA(), logtA, graph);
        AI_LOGGER(AI_SEVERITY_INFO, "Done.");

        return std::get<0>(a_v);
    }

    double MAUCEPolicy::getActionProbability(const Action & a) const {
        if (veccmp(a, sampleAction()) == 0) return 1.0;
        return 0.0;
    }

    const Experience & MAUCEPolicy::getExperience() const {
        return exp_;
    }
}
