#ifndef AI_TOOLBOX_POMDP_WITNESS_HEADER_FILE
#define AI_TOOLBOX_POMDP_WITNESS_HEADER_FILE

class Witness {
    // Qat(b) = SUM_s b(s) R(s,a) + gamma SUM_o Pr(O | a, b) V_{t-1}(b')
    // 
    // b' <= b, a, o
    // V_t(b) = max_a ( Qat(b) )
    // 
    // 
    // OUTER LOOP
    //
    // V_0 = { 0, 0, 0, 0, 0, ... }
    // t := 1
    // 
    // do {
    //      t++
    //      for ( a : A )
    //          Qat = witness(V_{t-1}, a)
    //      V_t = prune( union(Q:t) )
    // } while ( supremum_b(abs(V_t(b) - V_{t-1}(b))) < eps )
    //
    // 
    // INNER LOOP
    //
    // inputs: Ua, p_new
    // variables: delta, b
    // maximize: delta
    // improv constraints:  for each p in Ua: Vp_new(b) - Vp(b) >= delta
    // simplex constraints: for each s in S:  b(s) >= 0;  sum(b) = 1.0
    //
};

#endif
