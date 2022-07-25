#include <AIToolbox/Factored/Bandit/Algorithms/Utils/MaxPlus.hpp>

#include <AIToolbox/Factored/Bandit/Algorithms/Utils/LocalSearch.hpp>
#include <AIToolbox/Logging.hpp>

namespace AIToolbox::Factored::Bandit {
    MaxPlus::MaxPlus(const unsigned iterations) : iterations_(iterations) {}

    MaxPlus::Result MaxPlus::operator()(const Action & A, const Graph & graph) {
        // Preallocate memory.
        // - retval keeps the best currently found solution.
        // - bestCurrent keeps the currently found solution.
        Result retval, bestCurrent;

        auto & [rAction, rValue] = retval;
        auto & [cAction, cValue] = bestCurrent;

        rAction.resize(A.size());
        cAction.resize(A.size());

        rValue = std::numeric_limits<double>::lowest();

        // Initialize the message caches.
        // - inMessages are the previous timestep's messages sent
        // to agents, which need to be read by the factors.
        // - outMessages are the messages outbound from the factors
        // to the agent.
        // We need both to avoid overwriting old messages during a
        // single message-passing step.
        std::vector<Matrix2D> inMessages(A.size()), outMessages(A.size());
        for (size_t a = 0; a < A.size(); ++a) {
            const auto rows = graph.getFactors(a).size();
            // Here each row corresponds to the message from
            // a factor to this agent.
            // We add an additional row to store the sum of all
            // rows, to avoid having to re-sum the whole matrix
            // all the time.
            inMessages[a].resize(rows + 1, A[a]);
            outMessages[a].resize(rows + 1, A[a]);

            // We don't need to zero inMessages, we do it at the start
            // of each message passing iteration, just after the swap.
            outMessages[a].setZero();
        }

        // Initialize temporary local factor messages with max possible sizes.
        size_t maxRows = 0, maxCols = 0;
        for (auto f = graph.begin(); f != graph.end(); ++f) {
            maxRows = std::max(maxRows, f->getVariables().size());
            maxCols = std::max(maxCols, static_cast<size_t>(f->getData().size()));
        }
        // This stores the messages the factor will send to its adjacent
        // agents, as they are being constructed.
        Matrix2D factorMessage(maxRows + 1, maxCols);

        for (size_t iters = 0; iters < iterations_; ++iters) {
            // Since we have processed outMessages in the previous iteration
            // step, we can now swap in&out, and reset the out for this
            // iteration.
            std::swap(inMessages, outMessages);
            for (auto & m : outMessages)
                m.setZero();

            AI_LOGGER(AI_SEVERITY_DEBUG, "MaxPlus: iteration " << iters + 1);

            for (auto f = graph.begin(); f != graph.end(); ++f) {
                const auto & aNeighbors = graph.getVariables(f);

                // For each factor in the graph, we compute its unmaximized
                // message by summing its original function with the
                // appropriate messages from its adjacent agents.
                //
                // Note: we use head to avoid Eigen reallocating memory.
                // message is also only accessed by indexes (and we never take
                // its size directly), so it should be ok.
                const size_t aSize = aNeighbors.size();
                const size_t fSize = f->getData().size();
                factorMessage.row(aSize).head(fSize) = f->getData();

                size_t len = 1;
                for (size_t ai = 0; ai < aSize; ++ai) {
                    const auto a = aNeighbors[ai];
                    // Figure out the ID of this factor w.r.t. the current agent
                    // so we know which row of messages to read.
                    const auto & fNeighbors = graph.getFactors(a);
                    const auto fId = std::distance(std::begin(fNeighbors), std::find(std::begin(fNeighbors), std::end(fNeighbors), f));

                    const auto bottomRowId = inMessages[a].rows() - 1;

                    // Remove from sum the message coming from this factor
                    inMessages[a].row(bottomRowId) -= inMessages[a].row(fId);

                    // Add each element of the message in the correct place for the
                    // cross-sum across all agents. This code is basically equivalent to
                    //
                    //     message += np.tile(np.repeat(inMessage, len), ...)
                    size_t i = 0;
                    while (i < fSize)
                        for (size_t j = 0; j < A[a]; ++j)
                            for (size_t l = 0; l < len; ++l)
                                factorMessage.row(ai)[i++] = inMessages[a].row(bottomRowId)[j];

                    // Restore agent's sum message for later
                    inMessages[a].row(bottomRowId) += inMessages[a].row(fId);

                    len *= A[a];

                    // Add message from this agent to the global sum as well
                    factorMessage.row(aSize).head(fSize) += factorMessage.row(ai).head(fSize);
                }

                // Once the overall message is computed, we selectively
                // maximize over it depending on which agent we are sending the
                // message to (we maximize all other agents).
                for (size_t ai = 0; ai < aSize; ++ai) {
                    const auto a = aNeighbors[ai];
                    const auto & fNeighbors = graph.getFactors(a);
                    const auto fId = std::distance(std::begin(fNeighbors), std::find(std::begin(fNeighbors), std::end(fNeighbors), f));

                    // Remove message from this agent from the global sum
                    factorMessage.row(aSize).head(fSize) -= factorMessage.row(ai).head(fSize);

                    // Compute the out message for each action of this agent.
                    double norm = 0.0;
                    for (size_t av = 0; av < A[a]; ++av) {
                        // Here we list all joint-action ids where the action
                        // of agent 'a' is equal to 'av'. We use them to access
                        // the appropriate 'message' values, and we maximize
                        // over them into 'outMessage'.
                        PartialIndexEnumerator e(A, aNeighbors, a, av);

                        double outMessage = std::numeric_limits<double>::lowest();
                        while (e.isValid()) {
                            outMessage = std::max(outMessage, factorMessage(aSize, *e));
                            e.advance();
                        }
                        outMessages[a](fId, av) = outMessage;
                        norm += outMessage;
                    }

                    // Add back message from this agent from the global sum
                    factorMessage.row(aSize).head(fSize) += factorMessage.row(ai).head(fSize);

                    // Finally, we normalize the message (from the MaxPlus
                    // paper). This is done to avoid value explosions in loopy
                    // graphs (as a factor's messages will eventually come back
                    // to it over the loops and be summed infinitely).
                    outMessages[a].row(fId).array() -= norm / A[a];
                }
            }

            // Finally check whether we have a new best action, and
            // compute the messages for the next iteration.
            //
            // Here we also handle the agent nodes' part of the work. We sum
            // all messages received in the last row of the outMessages
            // (reserved for this purpose), and use it to see if we have found
            // a better joint action.
            //
            // We still keep the rest of the matrix so it's easier to compute
            // the inMessages next iteration (one subtraction for each factor,
            // rather than re-summing the matrix every time).
            for (size_t a = 0; a < A.size(); ++a) {
                auto & m = outMessages[a];
                // Also last row ID
                const auto rowsMinusOne = m.rows() - 1;

                // Compute "generic" message, we subtract each row one
                // at a time when it is needed. Note that we need to
                // avoid summing the last row since we have
                // "incorrectly" subtracted the normalization from it.
                m.row(rowsMinusOne) = m.topRows(rowsMinusOne).colwise().sum();

                // Compute the local best action for this agent (and its value,
                // which would ideally be the overall joint action value *for
                // all agents*).
                //
                // Note that we do not save up the value of the action here,
                // since it won't really add up to the true value of the action
                // (which we compute later).
                m.row(rowsMinusOne).maxCoeff(&cAction[a]);
            }

            // If we need to evaluate the same action as before, just keep
            // iterating.
            // Ideally we'd be able to detect convergence, but I'm not sure how.
            if (cAction == rAction) continue;

            // Compute true value of the current action to see whether we
            // should replace our current best with it.
            cValue = LocalSearch::evaluateGraph(A, graph, cAction);

            // We only change the selected action if it improves on the
            // previous best value.
            if (cValue > rValue)
                retval = bestCurrent;
        }
        // It can happen that the initial default action is indeed the best. If
        // that's the case, we'll never have a chance to update its true value,
        // so we do it here if it is still needed.
        if (rValue == std::numeric_limits<double>::lowest())
            rValue = LocalSearch::evaluateGraph(A, graph, rAction);

        return retval;
    }

    unsigned MaxPlus::getIterations() const { return iterations_; }
    void MaxPlus::setIterations(const unsigned iterations) { iterations_ = iterations; }
}
