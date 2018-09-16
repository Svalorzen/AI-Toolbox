/* This file contains the code that is presented in the Tutorial.md
 * file in the main folder of this project. The tutorial can also
 * be viewed in the main page of the doxygen documentation.
 *
 * This code implements a problem where a tiger needs to plan
 * in order to catch an antelope in a NxN toroidal grid world.
 *
 * The implementation is not efficient since all transition
 * probabilities are computed on the fly; storing them in a
 * matrix would make the solver work faster. Modifying the code
 * to allow this is trivial, and I wanted to keep the example
 * simple in order to introduce the theory behind the library.
 *
 * For more examples be sure to check out the "tests" folder!
 * The code there is very simple and it contains most usages
 * of this library ever, and it will probably give you an
 * even better introduction than this code does.
 */
#include <iostream>
#include <iomanip>
#include <fstream>
#include <array>
#include <cmath>
#include <chrono>
#include <thread>

#include <AIToolbox/MDP/Algorithms/ValueIteration.hpp>
#include <AIToolbox/MDP/Policies/Policy.hpp>
#include <AIToolbox/MDP/IO.hpp>
#include <AIToolbox/MDP/SparseModel.hpp>

// MODEL

constexpr int SQUARE_SIZE = 8;

using CoordType = std::array<int, 4>;
enum {
    TIGER_X = 0,
    TIGER_Y = 1,
    ANTEL_X = 2,
    ANTEL_Y = 3
};

// Returns distance between coordinates. It is consistent with
// the wraparound world.
int wrapDiff( int coord1, int coord2 ) {
    int diff = coord2 - coord1;

    int distance1 = std::abs( diff ), distance2 = SQUARE_SIZE - distance1;
    if ( distance1 < distance2 ) return diff;
    return diff > 0 ? -distance2 : distance2;
}

size_t A = 5;
enum {
    UP    = 0,
    DOWN  = 1,
    LEFT  = 2,
    RIGHT = 3,
    STAND = 4
};

double getTransitionProbability( const CoordType & c1, size_t action, const CoordType & c2 ) {
    // We compute the distances traveled by both the antelope and the tiger.
    int tigerMovementX = wrapDiff( c1[TIGER_X], c2[TIGER_X] );
    int tigerMovementY = wrapDiff( c1[TIGER_Y], c2[TIGER_Y] );
    int antelMovementX = wrapDiff( c1[ANTEL_X], c2[ANTEL_X] );
    int antelMovementY = wrapDiff( c1[ANTEL_Y], c2[ANTEL_Y] );

    // Both the tiger and the antelope can only move by 1 cell max at each
    // timestep. Thus, if this is not the case, the transition is
    // impossible.
    if ( std::abs( tigerMovementX ) +
         std::abs( tigerMovementY ) > 1 ) return 0.0;

    if ( std::abs( antelMovementX ) +
         std::abs( antelMovementY ) > 1 ) return 0.0;

    // Now we check whether the tiger was next to the antelope or not
    int diffX = wrapDiff( c1[TIGER_X], c1[ANTEL_X] );
    int diffY = wrapDiff( c1[TIGER_Y], c1[ANTEL_Y] );

    // We check whether they were both in the same cell before.
    // In that case the game would have ended, and nothing would happen anymore.
    // We model this as a self-absorbing state, or a state that always transitions
    // to itself. This is valid no matter the action taken.
    if ( diffX == 0 && diffY == 0 ) {
        if ( c1 == c2 ) return 1.0;
        else return 0.0;
    }

    // The tiger can move only in the direction specified by its action. If
    // it is not the case, the transition is impossible.
    if ( action == STAND && ( tigerMovementX || tigerMovementY ) ) return 0.0;
    if ( action == UP    && tigerMovementY != 1  ) return 0.0;
    if ( action == DOWN  && tigerMovementY != -1 ) return 0.0;
    if ( action == LEFT  && tigerMovementX != -1 ) return 0.0;
    if ( action == RIGHT && tigerMovementX != 1  ) return 0.0;

    // If they were not adjacent, then the probability for any move of the
    // antelope is simply 1/5: it behaves randomly.
    if ( std::abs( diffX ) + std::abs( diffY ) > 1 ) return 1.0 / 5.0;

    // Otherwise, first we check that the move was allowed, as
    // the antelope cannot move where the tiger was before.
    if ( c1[TIGER_X] == c2[ANTEL_X] && c1[TIGER_Y] == c2[ANTEL_Y] ) return 0.0;

    // Else the probability of this transition is 1 / 4, still random but without
    // a possible antelope action.
    return 1.0 / 4.0;
}

double getReward( const CoordType & c ) {
    if ( c[TIGER_X] == c[ANTEL_X] && c[TIGER_Y] == c[ANTEL_Y] ) return 10.0;
    return 0.0;
}

constexpr double discount = 0.9;

size_t encodeState(const CoordType & coords) {
    size_t state = 0; unsigned multiplier = 1;
    for ( auto c : coords ) {
        state += multiplier * c;
        multiplier *= SQUARE_SIZE;
    }
    return state;
}

CoordType decodeState(size_t state) {
    CoordType coords;
    for ( auto & c : coords ) {
        c = state % SQUARE_SIZE;
        state /= SQUARE_SIZE;
    }
    return coords;
}

class GridWorld {
    public:
        size_t getS() const { return SQUARE_SIZE * SQUARE_SIZE * SQUARE_SIZE * SQUARE_SIZE; }
        size_t getA() const { return ::A; }
        double getDiscount() const { return ::discount; }

        double getTransitionProbability( size_t s, size_t a, size_t s1 ) const {
            return ::getTransitionProbability( decodeState( s ), a, decodeState( s1 ) );
        }

        double getExpectedReward( size_t, size_t, size_t s1 ) const {
            return ::getReward( decodeState( s1 ) );
        }

        // These two functions are needed to keep template code in the library
        // simple, but you don't need to implement them for the method we use
        // in this example.
        std::tuple<size_t, double> sampleSR(size_t,size_t) const;
        bool isTerminal(size_t) const;
};

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

void printState(const CoordType & c) {
    for ( int y = SQUARE_SIZE - 1; y >= 0; --y ) {
        for ( int x = 0; x < SQUARE_SIZE; ++x ) {
            if (x == c[TIGER_X] && y == c[TIGER_Y]) std::cout << "@ ";
            else if (x == c[ANTEL_X] && y == c[ANTEL_Y]) std::cout << "A ";
            else std::cout << ". ";
        }
        std::cout << std::endl;
    }
}

void printCurrentTimeString() {
    auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    auto tm = *std::localtime(&t);
    std::cout << std::put_time(&tm, "%T");
}

int main() {
    GridWorld world;

    // This is optional, and should make solving the model almost instantaneous.
    // Unfortunately, since our main model is so big, the copying process
    // still takes a lot of time. But at least that would be a one-time cost!
    printCurrentTimeString(); std::cout << " - Constructing MDP...\n";
    AIToolbox::MDP::SparseModel model(world);

    // This is a method that solves MDPs completely. It has a couple of
    // parameters available.
    // The only non-optional parameter is the horizon of the solution; as in
    // how many steps should the solution look ahead in order to decide which
    // move to take. If we chose 1, for example, the tiger would only consider
    // cells next to it to decide where to move; this wouldn't probably be
    // what we want.
    // We want the tiger to think for infinite steps: this can be
    // approximated with a very high horizon, since in theory the final solution
    // will converge to a single policy anyway. Thus we put a very high number
    // as the horizon here.
    printCurrentTimeString(); std::cout << " - Solving MDP using infinite horizon...\n";
    AIToolbox::MDP::ValueIteration solver(1000000);

    // This is where the magic happen. This could take around 10-20 minutes,
    // depending on your machine (most of the time is spent on this tutorial's
    // code, however, since it is a pretty inefficient implementation).
    // But you can play with it and make it better!
    //
    // If you are using the Sparse Model though, it is instantaneous since
    // Eigen is very very efficient in computing the values we need!
    auto solution = solver(model);

    printCurrentTimeString();
    std::cout << " - Converged: " << (std::get<0>(solution) < solver.getTolerance()) << "\n";

    AIToolbox::MDP::Policy policy(world.getS(), world.getA(), std::get<1>(solution));

    // We create a random engine to pick a random state as start.
    std::default_random_engine rand(AIToolbox::Impl::Seeder::getSeed());
    std::uniform_int_distribution<size_t> start(0, SQUARE_SIZE * SQUARE_SIZE * SQUARE_SIZE * SQUARE_SIZE - 1);

    size_t s, a, s1;
    double r, totalReward = 0.0;

    // We create a starting state which is not the end.
    do s = start(rand);
    while (model.isTerminal(s));

    size_t t = 100;
    while (true) {
        // Print it!
        printState(decodeState(s));

        // We give the tiger a time limit, but if it reaches
        // the antelope we end the game.
        if (t == 0 || model.isTerminal(s)) break;

        // We sample an action for this state according to the optimal policy
        a = policy.sampleAction(s);
        // And we use the model to simulate what is going to happen next (in
        // case of a "real world" scenario where the library is used this step
        // would not exist as the world would automatically step to the next
        // state. Here we simulate.
        std::tie(s1, r) = model.sampleSR(s, a);

        // Add into the total reward (we don't use this here, it's just as an
        // example)
        totalReward += r;
        // Update the current state with the new one.
        s = s1;

        --t;
        goup(SQUARE_SIZE);

        // Sleep 1 second so the user can see what is happening.
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // You can save, and then load up this policy again on files. You will not
    // need to solve the model again ever, and you can embed the policy into
    // any application you want!
    // {
    //     std::ofstream output("policy.txt");
    //     output << policy;
    // }

    return 0;
}
