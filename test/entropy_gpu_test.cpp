/// Parity tests for the two GPU implementations of entropy::gpu::nonparametric::entries_dS.
///
/// Both live behind `#pragma omp declare target`, which makes them callable from the host, so these
/// tests exercise the real device code without needing a GPU. The implementation is selected by
/// whether the CSR carries the neighbour-block companions, so the same graph can be evaluated both
/// ways by passing a shallow CSR copy with those two pointers nulled out.
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include <omp.h>

#include "args.hpp"
#include "blockmodel.hpp"
#include "entropy.hpp"
#include "finetune.hpp"
#include "graph.hpp"
#include "typedefs.hpp"
#include "utils.hpp"

namespace {

/// Builds a Graph from an edge list, sized so that vertices with no edges are still represented.
Graph build_graph(const std::vector<std::pair<long, long>> &edges, long num_vertices) {
    NeighborList out_neighbors(num_vertices);
    NeighborList in_neighbors(num_vertices);
    std::vector<bool> self_edges(num_vertices, false);
    for (const auto &edge : edges) {
        utils::insert(out_neighbors, edge.first, edge.second);
        utils::insert(in_neighbors, edge.second, edge.first);
        if (edge.first == edge.second) self_edges[edge.first] = true;
    }
    std::vector<long> assignment(num_vertices, 0);
    return Graph(out_neighbors, in_neighbors, num_vertices, (long) edges.size(), self_edges, assignment);
}

/// The 11-vertex toy graph, which has self-loops on vertices 0, 5 and 10.
std::vector<std::pair<long, long>> toy_edges() {
    return {{0, 0}, {0, 1}, {0, 2}, {1, 2}, {2, 3}, {3, 1}, {3, 2}, {3, 5},
            {4, 1}, {4, 6}, {5, 4}, {5, 5}, {5, 6}, {5, 7}, {6, 4}, {7, 3},
            {7, 9}, {8, 5}, {8, 7}, {9, 10}, {10, 7}, {10, 8}, {10, 10}};
}

class EntropyGPUTest : public ::testing::Test {
protected:
    void SetUp() override {
        saved_algorithm = args.algorithm;
        saved_sparse = args.sparse_entries_ds;
        saved_csrgraph = args.csrgraph;
        saved_matrix_type = args.matrix_type;
        saved_parametric = args.parametric;
        saved_noduplicates = args.noduplicates;
        saved_degreecorrected = args.degreecorrected;
        saved_undirected = args.undirected;
        // The companions are only allocated for this algorithm with the flag set.
        args.algorithm = "async_gibbs_gpu";
        args.sparse_entries_ds = true;
        args.csrgraph = true;
        args.matrix_type = "dense";  // required by Blockmodel::gpu_view()
        args.parametric = false;
        args.noduplicates = false;   // keep parallel edges: they produce multi-edge runs
        args.degreecorrected = false;
        args.undirected = false;
        args.threads = 1;
        // The dense sweep reduces over an OpenMP parallel for; pinning to one thread keeps its
        // summation order fixed so the comparison below is not chasing reduction-order noise.
        omp_set_num_threads(1);
    }

    void TearDown() override {
        args.algorithm = saved_algorithm;
        args.sparse_entries_ds = saved_sparse;
        args.csrgraph = saved_csrgraph;
        args.matrix_type = saved_matrix_type;
        args.parametric = saved_parametric;
        args.noduplicates = saved_noduplicates;
        args.degreecorrected = saved_degreecorrected;
        args.undirected = saved_undirected;
    }

    /// Evaluates every (vertex, proposed block) pair both ways and asserts the two agree.
    static void expect_parity(const Graph &graph, const Blockmodel &blockmodel) {
        BlockmodelGPUView view = blockmodel.gpu_view();
        const CSR &csr = graph.out_csr();
        const CSR &csc = graph.in_csr();
        ASSERT_NE(csr._block_id, nullptr) << "companions were not allocated; the sparse path "
                                             "would not be exercised";

        for (long vertex = 0; vertex < graph.num_vertices(); ++vertex)
            finetune::gpu::refresh_neighbor_blocks(vertex, csr, csc, view);

        // CSR is a non-owning POD, so a shallow copy with the companions hidden selects the sweep
        // over exactly the same adjacency data.
        CSR csr_sweep = csr;
        csr_sweep._block_id = nullptr;
        csr_sweep._edge_weight = nullptr;
        CSR csc_sweep = csc;
        csc_sweep._block_id = nullptr;
        csc_sweep._edge_weight = nullptr;

        for (long vertex = 0; vertex < graph.num_vertices(); ++vertex) {
            long current_block = blockmodel.block_assignment(vertex);
            for (long proposed_block = 0; proposed_block < blockmodel.num_blocks(); ++proposed_block) {
                if (proposed_block == current_block) continue;
                ProposedMove proposal { vertex, current_block, proposed_block,
                                        csr.degree(vertex), csc.degree(vertex),
                                        csr.degree(vertex) + csc.degree(vertex) };
                double sparse = entropy::gpu::nonparametric::entries_dS(view, csr, csc, proposal);
                double sweep = entropy::gpu::nonparametric::entries_dS(view, csr_sweep, csc_sweep, proposal);
                EXPECT_NEAR(sweep, sparse, 1e-9 * std::max(1.0, std::fabs(sweep)))
                        << "entries_dS disagreed for vertex " << vertex
                        << " (block " << current_block << ") -> block " << proposed_block;

                double sparse_mdl = entropy::gpu::nonparametric::delta_mdl(view, csr, csc, proposal);
                double sweep_mdl = entropy::gpu::nonparametric::delta_mdl(view, csr_sweep, csc_sweep, proposal);
                EXPECT_NEAR(sweep_mdl, sparse_mdl, 1e-9 * std::max(1.0, std::fabs(sweep_mdl)))
                        << "delta_mdl disagreed for vertex " << vertex
                        << " (block " << current_block << ") -> block " << proposed_block;
            }
        }
    }

    std::string saved_algorithm, saved_matrix_type;
    bool saved_sparse = false, saved_csrgraph = false, saved_parametric = false;
    bool saved_noduplicates = false, saved_degreecorrected = false, saved_undirected = false;
};

TEST_F(EntropyGPUTest, SparseAndSweepEntriesDSAgreeOnToyGraph) {
    Graph graph = build_graph(toy_edges(), 11);
    std::vector<long> assignment = {0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2};
    Blockmodel blockmodel(3, graph, 0.5, assignment);
    expect_parity(graph, blockmodel);
}

TEST_F(EntropyGPUTest, SparseAndSweepEntriesDSAgreeWhenEveryVertexIsItsOwnBlock) {
    // Maximises the number of distinct neighbour blocks, so most runs have length one.
    Graph graph = build_graph(toy_edges(), 11);
    std::vector<long> assignment = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    Blockmodel blockmodel(11, graph, 0.5, assignment);
    expect_parity(graph, blockmodel);
}

TEST_F(EntropyGPUTest, SparseAndSweepEntriesDSAgreeOnGraphWithIsolatedVertices) {
    // Vertices 3 and 4 have no edges at all, which drives the size-0 path through the sort and
    // through both run-length loops.
    std::vector<std::pair<long, long>> edges = {{0, 0}, {0, 1}, {1, 2}, {2, 0}, {2, 2}, {5, 1}, {1, 5}};
    Graph graph = build_graph(edges, 6);
    std::vector<long> assignment = {0, 1, 0, 1, 0, 1};
    Blockmodel blockmodel(2, graph, 0.5, assignment);
    expect_parity(graph, blockmodel);
}

TEST_F(EntropyGPUTest, SparseAndSweepEntriesDSAgreeOnRandomGraph) {
    const long num_vertices = 120;
    const long num_blocks = 8;
    std::mt19937 rng(12345);
    std::uniform_int_distribution<long> vertex_dist(0, num_vertices - 1);

    // Parallel edges and self-loops are both left in, so runs of length > 1 and the self-edge
    // corrections are both exercised.
    std::vector<std::pair<long, long>> edges;
    for (long i = 0; i < 700; ++i)
        edges.emplace_back(vertex_dist(rng), vertex_dist(rng));
    Graph graph = build_graph(edges, num_vertices);

    std::vector<long> assignment(num_vertices);
    for (long v = 0; v < num_vertices; ++v) assignment[v] = v % num_blocks;  // no empty blocks
    std::shuffle(assignment.begin(), assignment.end(), rng);
    Blockmodel blockmodel(num_blocks, graph, 0.5, assignment);
    expect_parity(graph, blockmodel);
}

TEST(HeapSortTest, SortsKeysAscendingAndCarriesValues) {
    std::mt19937 rng(999);
    std::uniform_int_distribution<long> key_dist(0, 12);  // small range forces duplicate keys
    for (long size = 0; size <= 40; ++size) {
        std::vector<long> keys(size), values(size);
        for (long i = 0; i < size; ++i) {
            keys[i] = key_dist(rng);
            values[i] = i;
        }
        std::vector<std::pair<long, long>> expected;
        for (long i = 0; i < size; ++i) expected.emplace_back(keys[i], values[i]);
        std::sort(expected.begin(), expected.end());

        finetune::gpu::heap_sort(keys.data(), values.data(), size);

        EXPECT_TRUE(std::is_sorted(keys.begin(), keys.end())) << "size " << size;
        // Heap sort is not stable, so compare the (key, value) multiset rather than the order.
        std::vector<std::pair<long, long>> actual;
        for (long i = 0; i < size; ++i) actual.emplace_back(keys[i], values[i]);
        std::sort(actual.begin(), actual.end());
        EXPECT_EQ(expected, actual) << "size " << size << ": values did not follow their keys";
    }
}

TEST(HeapSortTest, LeavesSurroundingMemoryUntouchedForEmptySlice) {
    // Degree-0 vertices hand heap_sort a zero-length slice; it must not step outside it.
    std::vector<long> keys = {7, 7, 7};
    std::vector<long> values = {1, 2, 3};
    finetune::gpu::heap_sort(keys.data() + 1, values.data() + 1, 0);
    EXPECT_EQ(keys, (std::vector<long>{7, 7, 7}));
    EXPECT_EQ(values, (std::vector<long>{1, 2, 3}));
}

}  // namespace
