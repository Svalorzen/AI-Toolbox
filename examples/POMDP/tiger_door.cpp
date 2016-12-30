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
    { R"(  __    __  )" },
    { R"(    /||\    )" },
    { R"(   / || \   )" },
    { R"(  /  ||  \  )" },
    { R"( /   ||   \ )" },
    { R"(     ||     )" },
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

    AIToolbox::Table3D transitions(boost::extents[S][A][S]);
    AIToolbox::Table3D rewards(boost::extents[S][A][S]);
    AIToolbox::Table3D observations(boost::extents[S][A][O]);

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
    // The 0.0 is the epsilon factor, used with high horizons. It gives a way
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
    auto a_id = policy.sampleAction(b, horizon);

    // Setup cout to pretty print the simulation.
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(6);

    // We loop for each step we have yet to do.
    for (int t = horizon - 1; t >= 0; --t) {
        auto currA = std::get<0>(a_id);
        auto s1_o_r = model.sampleSOR(s, currA);
        auto currO = std::get<1>(s1_o_r);
        { // Rendering of the environment, depends on state, action and observation.
            auto & left  = s ? prize : tiger;
            auto & right = s ? tiger : prize;
            for (size_t i = 0; i < prize.size(); ++i)
                std::cout << left[i] << hspacer << right[i] << '\n';

            auto & dleft  = currA == A_LEFT  ? openDoor : closedDoor;
            auto & dright = currA == A_RIGHT ? openDoor : closedDoor;
            for (size_t i = 0; i < prize.size(); ++i)
                std::cout << dleft[i] << hspacer << dright[i] << '\n';

            auto & sleft  = currA == A_LISTEN && currO == TIG_LEFT  ? sound : nosound;
            auto & sright = currA == A_LISTEN && currO == TIG_RIGHT ? sound : nosound;
            for (size_t i = 0; i < prize.size(); ++i)
                std::cout << sleft[i] << hspacer << sright[i] << '\n';

            std::cout << numspacer << b[0] << numspacer << hspacer
                      << numspacer << b[1] << '\n';

            for (const auto & m : man)
                std::cout << manhspacer << m << '\n';

            std::cout << "Timestep missing: " << t << "       \n";

            goup(3 * prize.size() + man.size() + 2);
        }
        // Now that we have rendered, we can use the observation to find out
        // what action we should do next. The new sampling depends on the
        // previous one since we're implicitly keeping track of the changing
        // belief.
        a_id = policy.sampleAction(std::get<1>(a_id), currO, t);

        // We also explicitly update the belief just to show the user what the
        // agent is thinking. This is not necessary, only for rendering
        // purpouses.
        b = AIToolbox::POMDP::updateBelief(model, b, currA, currO);
        // Then we update the world
        s = std::get<0>(s1_o_r);

        // Sleep 1 second so the user can see what is happening.
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    // Put the cursor back where it should be.
    godown(3 * prize.size() + man.size() + 2);

    return 0;
}
