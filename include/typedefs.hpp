/**
 * Useful type definitions and such.
 */
#ifndef SBP_TYPEDEFS_HPP
#define SBP_TYPEDEFS_HPP

#include <unordered_map>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tsl/robin_map.h"

/// Stores a list of edges for a given structure (vertex or block) and their weights. Only stores the second portion
/// of the edge, so additional information is needed to reconstruct the edge. i.e.: for a list of edges (1-->2), (1--4),
/// (1-->6), only [2, 4, 6] and the corresponding weights will be stored.
struct EdgeWeights {
    std::vector<long> indices;
    std::vector<long> values;

    void print() {
        if (this->indices.empty()) {
            std::cout << "[]" << std::endl;
            return;
        }
        std::cout << "[" << this->indices[0] << ": " << this->values[0] << ", ";
        for (size_t num_printed = 1; num_printed < this->indices.size() - 1; num_printed++) {
            if (num_printed % 25 == 0) {
                std::cout << std::endl << " ";
            }
            std::cout << this->indices[num_printed] << ": " << this->values[num_printed] << ", ";
        }
        std::cout << this->indices[this->indices.size() - 1] << ": " << this->values[this->indices.size() - 1] << "]" << std::endl;
    }
};

/// Stores intermediate evaluation info & timers for later printing.
struct PartialProfile {
    double iteration = -1;
    double mdl = -1;
    double normalized_mdl_v1 = -1;
    double modularity = -1;
    long mcmc_iterations = -1;
    double mcmc_time = 0.0;
    double mcmc_sequential_time = 0.0;
    double mcmc_parallel_time = 0.0;
    double mcmc_vertex_move_time = 0.0;
    ulong mcmc_moves = 0;
    double block_merge_time = 0.0;
    double block_merge_loop_time = 0.0;
    double block_split_time = 0.0;
    double block_split_loop_time = 0.0;
    double blockmodel_build_time = 0.0;
    double finetune_time = 0.0;
    double load_balancing_time = 0.0;
    double sort_time = 0.0;
    double access_time = 0.0;
    double update_assignment = 0.0;
    double total_time = 0.0;
    long num_blocks = 0;
};

/// Used to hash a pair of integers. Source: https://codeforces.com/blog/entry/21853
struct longPairHash {
    size_t operator() (const std::pair<long, long> &pair) const {
        return std::hash<long long>() (((long long) pair.first) ^ (((long long) pair.second) << 32));
    }
};

typedef std::vector<std::vector<long>> NeighborList;

template <typename T>
struct SparseVector {
    std::vector<long>    idx;   // The index of the corresponding element in data
    std::vector<T>      data;  // The non-zero values of the vector
    // /// Returns the sum of all elements in data.
    // inline T sum() {
    //     T result;
    //     for (const T &value: this->data) {
    //         result += value;
    //     }
    //     return result;
    // }
    inline SparseVector<T> operator/(const double &rhs) {
        SparseVector<T> result;
        for (long i = 0; i < this->idx.size(); ++i) {
            result.idx.push_back(this->idx[i]);
            result.data.push_back(this->data[i] / rhs); 
        }
        return result;
    }
};

//template <typename T>
//using MapVector = std::unordered_map<long, T>;

//template <typename T>
//using MapVector = absl::flat_hash_map<long, T>;

template <typename T>
using MapVector = tsl::robin_map<long, T>;
using LongEntry = std::pair<long, long>;

struct Merge {
    long block = -1;
    long proposal = -1;
    double delta_entropy = std::numeric_limits<double>::max();
};

struct Membership {
    long vertex = -1;
    long block = -1;
};

struct Vertex {
    long id;
    long out_degree;
    long in_degree;  // maybe add self-edge? that way, degree = out_degree + in_degree - self_edge..., but I don't think that we need total degree
};

#pragma omp begin declare target
/**
 * Non-owning view into a contiguous range of neighbor ids and their edge weights
 * (a row of a CSR matrix).
 *
 * Supports range-for (over indices), .size(), operator[], .empty(), val(i), and
 * to_vector() (free function, host-only, copies indices).
 * Lifetime is bound to the CSR object that owns the underlying arrays.
 *
 * The struct and all pure-pointer accessors are declared target so they can be
 * used inside #pragma omp target regions.  to_vector() is host-only.
 *
 * --- Weight semantics ---
 * `vals` is allowed to be nullptr (staging / NL-mode path has no weight array).
 * val(i) returns vals[i] when vals is set, otherwise 1.  This means unweighted
 * graphs and the staging code path produce weight 1 for every edge, identical to
 * the previous behaviour, without any additional branches in callers.
 *
 * The READ side is now weighted-ready.  To complete full weighted-graph support
 * the following ingest/storage work is still needed (in dependency order):
 *
 *  1. Ingest weights at source.  Extend the loaders (src/graph.cpp load_text /
 *     load_matrix_market, around lines 217, 299, 323) to read a third weight
 *     column when present.  MatrixMarket already has a value field; TSV needs a
 *     "from to weight" form, defaulting to 1 when absent.  Gate on an
 *     --weighted flag or auto-detect column count.
 *
 *  2. Carry weight through add_edge.  Change Graph::add_edge(long from, long to)
 *     (src/graph.cpp:105) to add_edge(long from, long to, long weight = 1).
 *     All existing callers keep compiling via the default.
 *
 *  3. Give the staging store somewhere to hold weights.  NeighborList
 *     (typedefs.hpp:69) is std::vector<std::vector<long>> — indices only.  Add
 *     parallel _out_staging_vals / _in_staging_vals of the same type to Graph,
 *     and have utils::insert push the weight in lockstep with the index.
 *
 *  4. Populate csr.vals for real.  In build_csr_matrix (src/graph.cpp:56)
 *     replace `csr.vals[pos] = 1` with the staged weight.  copy_csr already
 *     deep-copies vals, so the rule-of-five is fine.
 *
 *  5. Wire the NL-mode (non-csrgraph) staging path.  Once staging holds weights,
 *     update Graph::in_neighbors / out_neighbors (graph.hpp:82-83, 102-103) to
 *     call the 3-arg NeighborView ctor with the staged weight pointer instead of
 *     the 2-arg ctor, so val(i) returns real weights instead of the nullptr=>1
 *     fallback.
 *
 *  6. Audit weight-aware consumers.  degree() / num_edges() semantics (edge count
 *     vs. summed weight), modularity() (src/graph.cpp ~line 260, currently sets
 *     edge_weight = 1), and any sampling/blockmodel code that assumes unit edges.
 *     edge_weights() (src/finetune.cpp) is already correct after this task.
 *
 *  7. Type widths.  vals is `long`; fractional weights would require templating
 *     CSR / NeighborView / EdgeWeights.  Out of scope but worth planning early.
 */
struct NeighborView {
    /// Constructs an empty view (no indices, no weights).
    NeighborView() : ptr(nullptr), vals(nullptr), len(0) {}
    /// Constructs an index-only view (used by the NL-mode / staging path, which
    /// has no weight array; val(i) returns 1 for all i).
    NeighborView(const long* ptr, long len) : ptr(ptr), vals(nullptr), len(len) {}
    /// Constructs a view with both indices and weights (CSR path).
    NeighborView(const long* ptr, const long* vals, long len) : ptr(ptr), vals(vals), len(len) {}

    const long* begin() const { return ptr; }
    const long* end()   const { return ptr + len; }
    long size()         const { return len; }
    bool empty()        const { return len == 0; }
    const long& operator[](long i) const { return ptr[i]; }
    /// Returns the edge weight for the i-th neighbor.  Returns 1 if no weight
    /// array was provided (unweighted / staging path).
    long val(long i)    const { return vals ? vals[i] : 1; }

    const long* ptr;
    const long* vals;
    long len;
};

struct ProposedMove {
    long vertex;
    long current_block;
    long proposed_block;
    long num_out_neighbor_edges;
    long num_in_neighbor_edges;
    long num_neighbor_edges;
    // NeighborViews are intentionally NOT stored here: carrying their host
    // pointers by value through the GPU kernel forced a scratch spill that
    // faulted. Consumers re-fetch graph_csr/graph_csc.neighbors(vertex) instead.
};

const Vertex InvalidVertex { -1, 0, 0 };
#pragma omp end declare target

/// Copy a NeighborView into a new std::vector<long> (host-only).
inline std::vector<long> to_vector(const NeighborView &v) {
    return std::vector<long>(v.ptr, v.ptr + v.len);
}

//const Membership NullMembership { -1, -1 };

struct VertexMove {
    double delta_entropy;
    bool did_move;
    long vertex;
    long proposed_block;
};

struct VertexMove_v2 {
    double delta_entropy;
    bool did_move;
    long vertex;
    long proposed_block;
    EdgeWeights out_edges;
    EdgeWeights in_edges;
};

struct VertexMove_v3 {
    double delta_entropy;
    bool did_move;
    Vertex vertex;
    long proposed_block;
    EdgeWeights out_edges;
    EdgeWeights in_edges;
};

/// Used for GPU-based vertex moves. TODO: reduce memory usage by storing the minimum required information.
struct VertexMoveGPU {
    double delta_entropy;
    bool did_move;
    Vertex vertex;
    long proposed_block;
};

typedef std::unordered_map<std::pair<long, long>, long, longPairHash> PairIndexVector;

namespace map_vector {
/// Returns either 0, or the value stored in `vector[key]` if it exists.
    inline long get(const MapVector<long> &vector, long key) {
        const auto iterator = vector.find(key);
        if (iterator == vector.end())
            return 0;
        return iterator->second;
    }
}  // namespace map_vector

/// Returns either 0, or the value stored in `vector[key]` if it exists.
inline long get(const PairIndexVector &vector, const std::pair<long, long> &key) {
    const auto iterator = vector.find(key);
    if (iterator == vector.end())
        return 0;
    return iterator->second;
}

// template<class T>
// using SparseVector = std::vector
// typedef struct proposal_evaluation_t {
//     long proposed_block;
//     double delta_entropy;
// } ProposalEvaluation;

#endif // SBP_TYPEDEFS_HPP