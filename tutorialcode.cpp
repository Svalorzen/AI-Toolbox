#include <iostream>
#include <array>
#include <cmath>

constexpr int SQUARE_SIZE = 11;
size_t S = SQUARE_SIZE * SQUARE_SIZE * SQUARE_SIZE * SQUARE_SIZE; // Total number of states

using CoordType = std::array<int, 4>;
enum {
    TIGER_X = 0,
    TIGER_Y = 1,
    ANTEL_X = 2,
    ANTEL_Y = 3
};

// Returns -1 or +1 depending on the move. It is consistent with
// the wraparound world.
int wrapDiff( int coord1, int coord2 ) {
    int diff = coord2 - coord1;

    int distance1 = std::abs( diff ), distance2 = SQUARE_SIZE - distance1;
    if ( distance1 < distance2 ) return diff;
    return diff > 0 ? -distance2 : distance2;
}

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

size_t A = 5;

enum {
    UP    = 0,
    DOWN  = 1,
    LEFT  = 2,
    RIGHT = 3,
    STAND = 4
};

double getTransitionProbability( const CoordType & c1, size_t action, const CoordType & c2 ) {
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

    // The tiger can move only in the direction specified by its action. If
    // it is not the case, the transition is impossible.
    if ( action == STAND && ( tigerMovementX || tigerMovementY ) ) return 0.0;
    if ( action == UP    && tigerMovementY != 1  ) return 0.0;
    if ( action == DOWN  && tigerMovementY != -1 ) return 0.0;
    if ( action == LEFT  && tigerMovementX != -1 ) return 0.0;
    if ( action == RIGHT && tigerMovementX != 1  ) return 0.0;

    // Now we check whether the tiger was next to the antelope or not
    int diffX = wrapDiff( c1[TIGER_X], c1[ANTEL_X] );
    int diffY = wrapDiff( c1[TIGER_Y], c2[ANTEL_Y] );

    // If thew were not adjacent, then the probability for any move of the
    // antelope is simply 1/5: it behaves randomly.
    if ( std::abs( diffX ) + std::abs( diffY ) > 1 ) return 1.0 / 5.0;

    // Otherwise, first we check that the move was allowed, as
    // the antelope cannot move where the tiger was before.
    if ( c1[TIGER_X] == c2[ANTEL_X] && c1[TIGER_Y] == c2[ANTEL_Y] ) return 0.0;

    // As a last check, we check whether they were in the same position before.
    // In that case the game would have ended, and nothing would happen anymore.
    if ( c1[TIGER_X] == c2[ANTEL_X] && c1[TIGER_Y] == c2[ANTEL_Y] ) return 0.0;

    // Else the probability of this transition is 1 / 4, still random but without
    // a possible antelope action.
    return 1.0 / 4.0;
}
