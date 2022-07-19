#ifndef AI_TOOLBOX_FACTORED_APSP_HEADER_FILE
#define AI_TOOLBOX_FACTORED_APSP_HEADER_FILE

#include <AIToolbox/Factored/Types.hpp>
#include <AIToolbox/Factored/Utils/FactorGraph.hpp>

namespace AIToolbox::Factored {
    template <typename Factor>
    auto buildAdjacencyList(const Action & A, const FactorGraph<Factor> & graph);

    /**
     * @brief This function solves the APSP problem for the provided graph.
     *
     * This function computes the graph diameter; i.e. the shortest longest
     * path between any two variable nodes. Factor nodes are treated as
     * multi-edges for this purpose, so they do not count as actual nodes and
     * they do not (directly) contribute to the diameter size.
     *
     * This function can be used to compute the iteration parameter for
     * MaxPlus, as the number of message iterations needed should be the same
     * as the diameter of the graph.
     *
     * @param graph The graph to compute the diameter for.
     *
     * @return The diameter of the graph.
     */
    template <typename Factor>
    size_t APSP(const FactorGraph<Factor> & graph) {
        // We simply perform BFS on each node of the graph for its max distance,
        // return the min out of all.

        const auto A = graph.variableSize();
        size_t retval = 0;

        std::vector<size_t> distances(A);
        std::vector<size_t> front;
        front.reserve(A);

        const auto adjacencyList = buildAdjacencyList(graph);

        for (size_t a = 0; a < A; ++a) {
            std::fill(std::begin(distances), std::end(distances), 0);
            front.clear();

            front.push_back(a);
            distances[a] = A;

            for (size_t i = 0; i < front.size(); ++i) {
                const auto a2 = front[i];
                const auto d = (distances[a2] == A) ? 0 : distances[a2];

                for (auto n : adjacencyList[a2]) {
                    if (distances[n] == 0) {
                        distances[n] = d + 1;
                        front.push_back(n);
                    }
                }
            }

            distances[a] = 0;
            retval = std::max(retval, *std::max_element(std::begin(distances), std::end(distances)));
        }
        return retval;
    }

    /**
     * @brief This function computes an adjacency list between the variables of the input graph.
     *
     * This function returns a vector with one element for each variable. Each
     * variable's element is itself a vector containing the indeces of all
     * neighbors of the variable. Two variables are neighbors if they are
     * connected to at least one common factor.
     *
     * @param graph The graph to compute the adjacency list for.
     *
     * @return The adjacency list of the graph.
     */
    template <typename Factor>
    auto buildAdjacencyList(const FactorGraph<Factor> & graph) {
        const auto A = graph.variableSize();

        std::vector<std::vector<size_t>> adjacencyList;
        adjacencyList.resize(A);

        for (const auto & f : graph) {
            const auto & tag = f.getVariables();
            for (auto a : tag)
                adjacencyList[a].insert(std::end(adjacencyList[a]), std::begin(tag), std::end(tag));
        }

        for (size_t a = 0; a < A; ++a) {
            auto & al = adjacencyList[a];
            auto begin = std::begin(al), end = std::end(al);

            std::sort(begin, end);
            end = std::unique(begin, end);
            end = std::remove(begin, end, a);
            al.erase(end, std::end(al));
        }

        return adjacencyList;
    }
}

#endif
