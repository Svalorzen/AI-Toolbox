#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <thread>

#include <AIToolbox/POMDP/Model.hpp>
#include <AIToolbox/MDP/Model.hpp>

#include <AIToolbox/POMDP/Algorithms/IncrementalPruning.hpp>
#include <AIToolbox/POMDP/Policies/Policy.hpp>

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

const std::vector<std::string> prize {
    { R"(  ________  )" },
    { R"(  |       |\)" },
    { R"(  |_______|/)" },
    { R"( / $$$$  /| )" },
    { R"(+-------+ | )" },
    { R"(|       |/  )" },
    { R"(+-------+   )" }};

const std::vector<std::string> tiger {
    { R"(            )" },
    { R"(   (`/' ` | )" },
    { R"(  /'`\ \   |)" },
    { R"( /<7' ;  \ \)" },
    { R"(/  _､-, `,-\)" },
    { R"(`-`  ､/ ;   )" },
    { R"(     `-'    )" }};

const std::vector<std::string> closedDoor {
    { R"(   ______   )" },
    { R"(  /  ||  \  )" },
    { R"( |   ||   | )" },
    { R"( |   ||   | )" },
    { R"( |   ||   | )" },
    { R"( +===++===+ )" },
    { R"(            )" }};

const std::vector<std::string> openDoor {
    { R"(   ______   )" },
    { R"(|\/      \/|)" },
    { R"(||        ||)" },
    { R"(||        ||)" },
    { R"(||        ||)" },
    { R"(||________||)" },
    { R"(|/        \|)" }};

const std::vector<std::string> sound {
    { R"(    -..-    )" },
    { R"(            )" },
    { R"(  '-,__,-'  )" },
    { R"(            )" },
    { R"( `,_    _,` )" },
    { R"(    `--`    )" },
    { R"(            )" }};

const std::vector<std::string> nosound {
    { R"(            )" },
    { R"(            )" },
    { R"(            )" },
    { R"(            )" },
    { R"(            )" },
    { R"(            )" },
    { R"(            )" }};
// Different format for him!
const std::vector<std::string> man {
    { R"(   ___   )" },
    { R"(  //|\\  )" },
    { R"(  \___/  )" },
    { R"( \__|__/ )" },
    { R"(    |    )" },
    { R"(    |    )" },
    { R"(   / \   )" },
    { R"(  /   \  )" }};

// Random spaces to make the rendering look nice. Yeah this is ugly, but it's
// just for the rendering.
const std::string hspacer{"     "};
const std::string manhspacer(hspacer.size() / 2 + prize[0].size() - man[0].size() / 2, ' ');
const std::string numspacer((prize[0].size() - 8)/2, ' ');

const std::string clockSpacer = numspacer + std::string((hspacer.size() - 1) / 2, ' ');
const std::string strclock(R"(/|\-)");

// MODEL

enum {
    A_LISTEN = 0,
    A_LEFT   = 1,
    A_RIGHT  = 2,
};

enum {
    TIG_LEFT    = 0,
    TIG_RIGHT   = 1,
};

inline AIToolbox::POMDP::Model<AIToolbox::MDP::Model> makeTigerProblem() {
    // Actions are: 0-listen, 1-open-left, 2-open-right
    size_t S = 2, A = 3, O = 2;

    AIToolbox::POMDP::Model<AIToolbox::MDP::Model> model(O, S, A);

    AIToolbox::DumbMatrix3D transitions(boost::extents[S][A][S]);
    AIToolbox::DumbMatrix3D rewards(boost::extents[S][A][S]);
    AIToolbox::DumbMatrix3D observations(boost::extents[S][A][O]);

    // Transitions
    // If we listen, nothing changes.
    for ( size_t s = 0; s < S; ++s )
        transitions[s][A_LISTEN][s] = 1.0;

    // If we pick a door, tiger and treasure shuffle.
    for ( size_t s = 0; s < S; ++s ) {
        for ( size_t s1 = 0; s1 < S; ++s1 ) {
            transitions[s][A_LEFT ][s1] = 1.0 / S;
            transitions[s][A_RIGHT][s1] = 1.0 / S;
        }
    }

    // Observations
    // If we listen, we guess right 85% of the time.
    observations[TIG_LEFT ][A_LISTEN][TIG_LEFT ] = 0.85;
    observations[TIG_LEFT ][A_LISTEN][TIG_RIGHT] = 0.15;

    observations[TIG_RIGHT][A_LISTEN][TIG_RIGHT] = 0.85;
    observations[TIG_RIGHT][A_LISTEN][TIG_LEFT ] = 0.15;

    // Otherwise we get no information on the environment.
    for ( size_t s = 0; s < S; ++s ) {
        for ( size_t o = 0; o < O; ++o ) {
            observations[s][A_LEFT ][o] = 1.0 / O;
            observations[s][A_RIGHT][o] = 1.0 / O;
        }
    }

    // Rewards
    // Listening has a small penalty
    for ( size_t s = 0; s < S; ++s )
        for ( size_t s1 = 0; s1 < S; ++s1 )
            rewards[s][A_LISTEN][s1] = -1.0;

    // Treasure has a decent reward, and tiger a bad penalty.
    for ( size_t s1 = 0; s1 < S; ++s1 ) {
        rewards[TIG_RIGHT][A_LEFT][s1] = 10.0;
        rewards[TIG_LEFT ][A_LEFT][s1] = -100.0;

        rewards[TIG_LEFT ][A_RIGHT][s1] = 10.0;
        rewards[TIG_RIGHT][A_RIGHT][s1] = -100.0;
    }

    model.setTransitionFunction(transitions);
    model.setRewardFunction(rewards);
    model.setObservationFunction(observations);

    return model;
}

int main() {
    // We create a random engine, since we will need this later.
    std::default_random_engine rand(AIToolbox::Impl::Seeder::getSeed());

    // Create model of the problem.
    auto model = makeTigerProblem();
    model.setDiscount(0.95);

    // Set the horizon. This will determine the optimality of the policy
    // dependent on how many steps of observation/action we plan to do. 1 means
    // we're just going to do one thing only, and we're done. 2 means we get to
    // do a single action, observe the result, and act again. And so on.
    unsigned horizon = 15;
    // The 0.0 is the tolerance factor, used with high horizons. It gives a way
    // to stop the computation if the policy has converged to something static.
    AIToolbox::POMDP::IncrementalPruning solver(horizon, 0.0);

    // Solve the model. After this line, the problem has been completely
    // solved. All that remains is setting up an experiment and see what
    // happens!
    auto solution = solver(model);

    // We create a policy from the solution, in order to obtain actual actions
    // depending on what happens in the environment.
    AIToolbox::POMDP::Policy policy(2, 3, 2, std::get<1>(solution));

    // We begin a simulation, we start from a uniform belief, which means that
    // we have no idea on which side the tiger is in. We sample from the belief
    // in order to get a "real" state for the world, since this code has to
    // both emulate the environment and control the agent. The agent won't know
    // the sampled state though, it will only have the belief to work with.
    AIToolbox::POMDP::Belief b(2); b << 0.5, 0.5;
    auto s = AIToolbox::sampleProbability(2, b, rand);

    // The first thing that happens is that we take an action, so we sample it now.
    auto [a, ID] = policy.sampleAction(b, horizon);

    // Setup cout to pretty print the simulation.
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(6);

    // We loop for each step we have yet to do.
    double totalReward = 0.0;
    for (int t = horizon - 1; t >= 0; --t) {
        // We advance the world one step (the agent only sees the observation
        // and reward).
        auto [s1, o, r] = model.sampleSOR(s, a);
        // We and update our total reward.
        totalReward += r;

        { // Rendering of the environment, depends on state, action and observation.
            auto & left  = s ? prize : tiger;
            auto & right = s ? tiger : prize;
            for (size_t i = 0; i < prize.size(); ++i)
                std::cout << left[i] << hspacer << right[i] << '\n';

            auto & dleft  = a == A_LEFT  ? openDoor : closedDoor;
            auto & dright = a == A_RIGHT ? openDoor : closedDoor;
            for (size_t i = 0; i < prize.size(); ++i)
                std::cout << dleft[i] << hspacer << dright[i] << '\n';

            auto & sleft  = a == A_LISTEN && o == TIG_LEFT  ? sound : nosound;
            auto & sright = a == A_LISTEN && o == TIG_RIGHT ? sound : nosound;
            for (size_t i = 0; i < prize.size(); ++i)
                std::cout << sleft[i] << hspacer << sright[i] << '\n';

            std::cout << numspacer << b[0] << clockSpacer
                      << strclock[t % strclock.size()]
                      << clockSpacer << b[1] << '\n';

            for (const auto & m : man)
                std::cout << manhspacer << m << '\n';

            std::cout << "Timestep missing: " << t << "       \n";
            std::cout << "Total reward:     " << totalReward << "       " << std::endl;

            goup(3 * prize.size() + man.size() + 3);
        }

        // We explicitly update the belief to show the user what the agent is
        // thinking. This is also necessary in some cases (depending on
        // convergence of the solution, see below), otherwise its only for
        // rendering purpouses. It is a pretty expensive operation so if
        // performance is required it should be avoided.
        b = AIToolbox::POMDP::updateBelief(model, b, a, o);

        // Now that we have rendered, we can use the observation to find out
        // what action we should do next.
        //
        // Depending on whether the solution converged or not, we have to use
        // the policy differently. Suppose that we planned for an horizon of 5,
        // but the solution converged after 3. Then the policy will only be
        // usable with horizons of 3 or less. For higher horizons, the highest
        // step of the policy suffices (since it converged), but it will need a
        // manual belief update to know what to do.
        //
        // Otherwise, the policy implicitly tracks the belief via the id it
        // returned from the last sampling, without the need for a belief
        // update. This is a consequence of the fact that POMDP policies are
        // computed from a piecewise linear and convex value function, so
        // ranges of similar beliefs actually result in needing to do the same
        // thing (since they are similar enough for the timesteps considered).
        if (t > (int)policy.getH())
            std::tie(a, ID) = policy.sampleAction(b, policy.getH());
        else
            std::tie(a, ID) = policy.sampleAction(ID, o, t);

        // Then we update the world
        s = s1;

        // Sleep 1 second so the user can see what is happening.
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    // Put the cursor back where it should be.
    godown(3 * prize.size() + man.size() + 3);

    return 0;
}
