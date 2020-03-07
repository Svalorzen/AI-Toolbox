/* This file contains the code that is presented in the TutorialMDPRL.md file in
 * the docs folder. The tutorial can also be viewed in the main page of the
 * doxygen documentation.
 *
 * This code implements a problem where an agent needs to learn to walk on the
 * edge of a cliff without falling, to reach its destination.
 *
 * For more examples be sure to check out the "tests" folder! The code there
 * is very simple and it contains most usages of this library ever, and it will
 * probably give you an even better introduction than this code does.
 */
#include <iostream>
#include <chrono>
#include <thread>

#include <AIToolbox/MDP/Environments/CliffProblem.hpp>

#include <AIToolbox/MDP/Experience.hpp>
#include <AIToolbox/MDP/MaximumLikelihoodModel.hpp>

#include <AIToolbox/MDP/Algorithms/QLearning.hpp>
#include <AIToolbox/MDP/Algorithms/PrioritizedSweeping.hpp>

#include <AIToolbox/MDP/Policies/QGreedyPolicy.hpp>
#include <AIToolbox/MDP/Policies/EpsilonPolicy.hpp>

// RENDERING

// Special character to go back up when drawing.
std::string up =   "\033[XA";
// Special character to go back to the beginning of the line.
std::string back = "\33[2K\r";

void goup(unsigned x) {
    while (x > 9) {
        up[2] = '0' + 9;
        std::cout << up;
        x -= 9;
    }
    up[2] = '0' + x;
    std::cout << up;
}

void godown(unsigned x) {
    while (x) {
        std::cout << '\n';
        --x;
    }
}

// This prints the state knowing it's relative to the cliff problem.
void printState(size_t s, const AIToolbox::MDP::GridWorld & g) {
    for ( unsigned y = 0; y < g.getHeight(); ++y ) {
        for ( unsigned x = 0; x < g.getWidth(); ++x ) {
            // The highest values are for the corner of the cliffworld, the
            // height is actually one more than the grid knows.  See the
            // CliffProblem docs to understand exactly what the setup is.
            if (s >= g.getS()) {
                std::cout << ". ";
                continue;
            }
            auto c = g(s);
            if (x == c.getX() && (unsigned)y == c.getY()) std::cout << "@ ";
            else std::cout << ". ";
        }
        std::cout << std::endl;
    }
    // Now draw the cliff
    if (s == g.getS())
        std::cout << "@ ";
    else
        std::cout << ". ";

    for ( unsigned x = 0; x < g.getWidth() - 2; ++x )
        std::cout << "C ";

    if (s == g.getS() + 1)
        std::cout << "@ ";
    else
        std::cout << ". ";
    std::cout << std::endl;
}

AIToolbox::MDP::QFunction runQLearning(const AIToolbox::MDP::SparseModel & problem) {
    using namespace AIToolbox::MDP;

    std::cout << "Learning with QLearning...\n";

    // We create the QLearning method. It only needs to know the size of the
    // state and action space, and the discount of the problem (to correctly update
    // values).
    QLearning qlLearner(problem.getS(), problem.getA(), problem.getDiscount());

    // We get a reference to the QFunction that QLearning is updating, and use
    // it to construct a greedy policy.
    QGreedyPolicy gPolicy(qlLearner.getQFunction());

    // The greedy policy is then augmented with some randomness, to help the
    // agent explore. In this case, we are going to take random actions with
    // probability 0.1 (10%). In the other cases, we will ask the greedy policy
    // what to do, and return that.
    EpsilonPolicy ePolicy(gPolicy, 0.1);

    // Initial starting point, the bottom left corner.
    size_t start = problem.getS() - 2;

    std::cout << "Starting training...\n";

    size_t s, a;
    // We perform 10000 episodes, which should be enough to learn this problem.
    // At the start of each episode, we reset the position of the agent. Note
    // that this reset is for the episode; if during the episode the agent falls
    // into the cliff it will also be reset.
    for ( int episode = 0; episode < 10000; ++episode ) {
        s = start;
        // We limit the length of the episode to 10000 timesteps, to prevent the
        // agent roaming around indefinitely.
        for ( int i = 0; i < 10000; ++i ) {
            // Obtain an action for this state (10% random, 90% what we think is
            // best to do given the current QFunction).
            a = ePolicy.sampleAction( s );

            // Sample a new state and reward from the problem
            const auto [s1, rew] = problem.sampleSR( s, a );

            // Pass the newly collected data to QLearning, to update the
            // QFunction and improve the agent's policies.
            qlLearner.stepUpdateQ( s, a, s1, rew );

            // If we reach the goal, the episode ends
            if ( s1 == problem.getS() - 1 ) break;

            s = s1;
        }
    }
    std::cout << "Training over!\n";

    // Return a copy of the optimal QFunction
    return qlLearner.getQFunction();
}

AIToolbox::MDP::QFunction runPrioritizedSweeping(const AIToolbox::MDP::SparseModel & problem) {
    using namespace AIToolbox::MDP;

    std::cout << "Learning with PrioritizedSweeping...\n";

    std::cout << "Setting up Experience and MaximumLikelihoodModel...\n";

    // Create an Experience, to keep track of the transitions and rewards we
    // obtain during the interactions with the environment.
    Experience exp(problem.getS(), problem.getA());

    // Create a learned model to transform the data we have collected into
    // appropriate transition and reward functions, so we can reason about the
    // learned model and improve our learning.
    MaximumLikelihoodModel<Experience> learnedModel(exp, problem.getDiscount(), false);

    std::cout << "Setting up PrioritizedSweeping...\n";

    // Setup PrioritizedSweeping with the model we are learning.
    // PrioritizedSweeping will update a QFunction reflecting our best estimate
    // of the values of the problem.
    PrioritizedSweeping psLearner(learnedModel);

    // As in QLearning, setup two policies to select actions
    QGreedyPolicy gPolicy(psLearner.getQFunction());
    EpsilonPolicy ePolicy(gPolicy, 0.1);

    // Initial starting point, the bottom left corner.
    size_t start = problem.getS() - 2;

    std::cout << "Starting training...\n";

    size_t s, a;
    // We perform 100 episodes, which should be enough to learn this problem.
    // Note that PrioritizedSweeping needs much fewer episodes to learn
    // effectively, as it is using the learned model to extract as much
    // information as possible and doing many updates per timestep.
    // At the start of each episode, we reset the position of the agent. Note
    // that this reset is for the episode; if during the episode the agent falls
    // into the cliff it will also be reset.
    for ( int episode = 0; episode < 100; ++episode ) {
        s = start;
        // We limit the length of the episode to 10000 timesteps, to prevent the
        // agent roaming around indefinitely.
        for ( int i = 0; i < 10000; ++i ) {
            // Obtain an action for this state (10% random, 90% what we think is
            // best to do given the current QFunction).
            a = ePolicy.sampleAction( s );

            // Sample a new state and reward from the problem
            const auto [s1, rew] = problem.sampleSR( s, a );

            // Record the new data in the Experience, so we can track it
            exp.record(s, a, s1, rew);

            // Update the learned model with the data we have just got.
            // This updates both the transition and reward functions.
            learnedModel.sync(s, a, s1);

            // Update the QFunction using this data.
            psLearner.stepUpdateQ(s, a);
            // Finally, use PrioritizedSweeping reasoning capabilities in order
            // to perform additional updates, and learn much more rapidly that
            // QLearning.
            psLearner.batchUpdateQ();

            // If we reach the goal, the episode ends
            if ( s1 == problem.getS() - 1 ) break;

            s = s1;
        }
    }
    std::cout << "Training over!\n";

    // Return a copy of the optimal QFunction
    return psLearner.getQFunction();
}

int main(int argc, const char * argv[]) {
    bool useQL;
    bool printUsage = false;

    if (argc < 2)
        printUsage = true;
    else if (std::string(argv[1]) == "QL")
        useQL = true;
    else if (std::string(argv[1]) == "PS")
        useQL = false;
    else
        printUsage = true;

    if (printUsage) {
        std::cout << "Usage: " << argv[0] << " [QL|PS]\n";
        std::cout << "- Select QL to train QLearning\n";
        std::cout << "- Select PS to train PrioritizedSweeping\n";
        return 0;
    }

    using namespace AIToolbox::MDP;

    GridWorld grid(12, 3);

    // We then build a cliff problem out of it. The agent will start at the
    // bottom left corner of the grid, and its target will be the bottom right
    // corner. However, aside from this two corners, all the cells at the bottom
    // of the grid will be marked as the cliff: going there will give a large
    // penalty to the agent, and will reset its position to the bottom left corner.
    auto problem = makeCliffProblem(grid);

    QFunction qf;
    if (useQL)
        qf = runQLearning(problem);
    else
        qf = runPrioritizedSweeping(problem);

    // Make space for the visualization of the solution.
    std::cout << std::endl;

    // Now that we have an optimal QFunction, we can wrap a greedy policy
    // around it and see what happens.
    QGreedyPolicy gPolicy(qf);

    // Initial starting point, the bottom left corner.
    size_t s = problem.getS() - 2, a;

    // We limit the length of the episode to 10000 timesteps, to prevent the
    // agent roaming around indefinitely.
    for ( int i = 0; i < 10000; ++i ) {
        // We display the current state of the environment to check that the
        // policy is good.
        printState(s, grid);

        // If we reach the goal, the episode ends
        if ( s == problem.getS() - 1 ) break;

        // Obtain a greedy action for this state.
        a = gPolicy.sampleAction( s );

        // Sample a new state and reward from the problem
        const auto [s1, rew] = problem.sampleSR( s, a );
        (void)rew;

        s = s1;

        goup(4);

        // Sleep 1 second so the user can see what is happening.
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    return 0;
}
