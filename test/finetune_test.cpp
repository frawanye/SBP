#include <vector>

#include <gtest/gtest.h>

#include "blockmodel.hpp"
#include "blockmodel/delta.hpp"
#include "finetune.hpp"
#include "graph.hpp"
#include "globals.hpp"
#include "utils.hpp"

#include "toy_example.hpp"
#include "typedefs.hpp"

// TODO: figure out correct placement of these
//MPI_t mpi;  // Unused
//Args args;  // Unused

class FinetuneTest : public ToyExample {
protected:
    Blockmodel B3;
    void SetUp() override {
        ToyExample::SetUp();
        std::vector<long> assignment3 = { 0, 0, 0, 1, 2, 3, 3, 4, 5, 1, 5 };
        B3 = Blockmodel(6, graph, 0.5, assignment3);
//        Deltas[std::make_pair(0, 0)] = 1;
//        Deltas[std::make_pair(0, 1)] = 0;
//        Deltas[std::make_pair(0, 2)] = 1;
//        Deltas[std::make_pair(1, 0)] = 1;
//        Deltas[std::make_pair(1, 2)] = -1;
//        Deltas[std::make_pair(2, 0)] = 1;
//        Deltas[std::make_pair(2, 1)] = 0;
//        Deltas[std::make_pair(2, 2)] = -3;
    }
};

class FinetuneDenseTest : public ToyExample {
protected:
    Blockmodel B3;
    void SetUp() override {
        forced_matrix_type = "dense";
        ToyExample::SetUp();
        std::vector<long> assignment3 = { 0, 0, 0, 1, 2, 3, 3, 4, 5, 1, 5 };
        B3 = Blockmodel(6, graph, 0.5, assignment3);
    }
};

TEST_F(FinetuneTest, SetUpWorksCorrectly) {
    EXPECT_EQ(graph.num_vertices(), 11);
    EXPECT_EQ((long)graph.num_vertices(), graph.num_vertices());
    EXPECT_EQ(graph.num_vertices(), graph.num_vertices());
    EXPECT_EQ(graph.num_edges(), 23);
}

TEST_F(FinetuneTest, SparseEdgeCountUpdatesAreCorrect) {
    long vertex = 7;
    long current_block = B.block_assignment(vertex);
    EdgeWeights out_edges = finetune::edge_weights(graph.out_neighbors(vertex), vertex, false);
    EdgeWeights in_edges = finetune::edge_weights(graph.in_neighbors(vertex), vertex, true);
    EdgeWeights blocks_out_neighbors = finetune::block_edge_weights(B.block_assignment(), out_edges);
    EdgeWeights blocks_in_neighbors = finetune::block_edge_weights(B.block_assignment(), in_edges);
    SparseEdgeCountUpdates updates;
    finetune::edge_count_updates_sparse(B, 7, current_block, 0, out_edges, in_edges, updates);
    EXPECT_EQ(updates.block_row[0], 2);
    EXPECT_EQ(updates.block_row[1], 1);
    EXPECT_EQ(updates.block_row[2], 3);
    EXPECT_EQ(updates.block_col[0], 1);
    EXPECT_EQ(updates.block_col[2], 3);
    EXPECT_EQ(updates.proposal_row[0], 8);
    EXPECT_EQ(updates.proposal_row[1], 1);
    EXPECT_EQ(updates.proposal_row[2], 1);
    EXPECT_EQ(updates.proposal_col[0], 8);
    EXPECT_EQ(updates.proposal_col[1], 2);
    EXPECT_EQ(updates.proposal_col[2], 2);
}

TEST_F(FinetuneTest, SparseEdgeCountUpdatesWithSelfEdgesAreCorrect) {
    long vertex = 5;
    long current_block = B.block_assignment(vertex);
    EdgeWeights out_edges = finetune::edge_weights(graph.out_neighbors(vertex), vertex, false);
    EdgeWeights in_edges = finetune::edge_weights(graph.in_neighbors(vertex), vertex, true);
    EdgeWeights blocks_out_neighbors = finetune::block_edge_weights(B.block_assignment(), out_edges);
    EdgeWeights blocks_in_neighbors = finetune::block_edge_weights(B.block_assignment(), in_edges);
    SparseEdgeCountUpdates updates;
    finetune::edge_count_updates_sparse(B, 5, current_block, 0, out_edges, in_edges, updates);
    EXPECT_EQ(updates.block_row[0], 1);
    EXPECT_EQ(updates.block_row[1], 2);
    EXPECT_EQ(updates.block_col[0], 2);
    EXPECT_EQ(updates.block_col[1], 2);
    EXPECT_EQ(updates.proposal_row[0], 9);
    EXPECT_EQ(updates.proposal_row[1], 2);
    EXPECT_EQ(updates.proposal_row[2], 1);
    EXPECT_EQ(updates.proposal_col[0], 9);
    EXPECT_EQ(updates.proposal_col[1], 1);
    EXPECT_EQ(updates.proposal_col[2], 2);
}

/// TODO: same test but using a vertex with a self edge
TEST_F(FinetuneTest, BlockmodelDeltasAreCorrect) {
    long vertex = 7;
    long current_block = B.block_assignment(vertex);
    EdgeWeights out_edges = finetune::edge_weights(graph.out_neighbors(vertex), vertex, false);
    EdgeWeights in_edges = finetune::edge_weights(graph.in_neighbors(vertex), vertex, false);
    Delta delta = finetune::blockmodel_delta(vertex, current_block, Proposal.proposal, out_edges, in_edges, B);
    EXPECT_EQ(delta.entries().size(), 6) << "blockmodel deltas are the wrong size. Expected 6 but got " << delta.entries().size();
    EXPECT_EQ(delta.get(0,0), 1);
    EXPECT_EQ(delta.get(0,1), 0);
    EXPECT_EQ(delta.get(0,2), 1);
    EXPECT_EQ(delta.get(1,0), 1);
    EXPECT_EQ(delta.get(1,2), -1);
    EXPECT_EQ(delta.get(2,0), 1);
    EXPECT_EQ(delta.get(2,1), 0);
    EXPECT_EQ(delta.get(2,2), -3);
}

/// Verifies that the COO Delta produces identical get() results to the map Delta
/// for the same vertex move proposal.
TEST_F(FinetuneTest, CooDeltaGetMatchesMapDelta) {
    long vertex = 7;
    long current_block = B.block_assignment(vertex);
    EdgeWeights out_edges = finetune::edge_weights(graph.out_neighbors(vertex), vertex, false);
    EdgeWeights in_edges = finetune::edge_weights(graph.in_neighbors(vertex), vertex, false);
    // Build both representations for the same move
    args.coodelta = false;
    Delta map_delta = finetune::blockmodel_delta(vertex, current_block, Proposal.proposal, out_edges, in_edges, B);
    args.coodelta = true;
    Delta coo_delta = finetune::blockmodel_delta(vertex, current_block, Proposal.proposal, out_edges, in_edges, B);
    args.coodelta = false;
    // get() must agree on the same known cells
    EXPECT_EQ(coo_delta.get(0, 0), map_delta.get(0, 0));
    EXPECT_EQ(coo_delta.get(0, 1), map_delta.get(0, 1));
    EXPECT_EQ(coo_delta.get(0, 2), map_delta.get(0, 2));
    EXPECT_EQ(coo_delta.get(1, 0), map_delta.get(1, 0));
    EXPECT_EQ(coo_delta.get(1, 2), map_delta.get(1, 2));
    EXPECT_EQ(coo_delta.get(2, 0), map_delta.get(2, 0));
    EXPECT_EQ(coo_delta.get(2, 1), map_delta.get(2, 1));
    EXPECT_EQ(coo_delta.get(2, 2), map_delta.get(2, 2));
    // Absolute values as a cross-check against the known answer
    EXPECT_EQ(coo_delta.get(0, 0),  1);
    EXPECT_EQ(coo_delta.get(0, 2),  1);
    EXPECT_EQ(coo_delta.get(1, 0),  1);
    EXPECT_EQ(coo_delta.get(1, 2), -1);
    EXPECT_EQ(coo_delta.get(2, 0),  1);
    EXPECT_EQ(coo_delta.get(2, 2), -3);
}

/// Verifies that the non-zero entries reported by the COO Delta match those of
/// the map Delta (after stripping zero-value entries, since COO may coalesce
/// cancelling add/sub pairs to zero while map mode retains them explicitly).
TEST_F(FinetuneTest, CooDeltaEntriesMatchMapDelta) {
    long vertex = 7;
    long current_block = B.block_assignment(vertex);
    EdgeWeights out_edges = finetune::edge_weights(graph.out_neighbors(vertex), vertex, false);
    EdgeWeights in_edges = finetune::edge_weights(graph.in_neighbors(vertex), vertex, false);
    args.coodelta = false;
    Delta map_delta = finetune::blockmodel_delta(vertex, current_block, Proposal.proposal, out_edges, in_edges, B);
    args.coodelta = true;
    Delta coo_delta = finetune::blockmodel_delta(vertex, current_block, Proposal.proposal, out_edges, in_edges, B);
    args.coodelta = false;
    // Build maps of (row,col)->val from each delta, filtering out zero-value entries
    std::map<std::pair<long,long>, long> map_vals, coo_vals;
    for (const auto &e : map_delta.entries()) {
        if (std::get<2>(e) != 0)
            map_vals[{std::get<0>(e), std::get<1>(e)}] = std::get<2>(e);
    }
    for (const auto &e : coo_delta.entries()) {
        if (std::get<2>(e) != 0)
            coo_vals[{std::get<0>(e), std::get<1>(e)}] = std::get<2>(e);
    }
    EXPECT_EQ(coo_vals, map_vals);
}

/// Same as CooDeltaGetMatchesMapDelta but for a vertex with a self-edge (vertex 10).
TEST_F(FinetuneTest, CooDeltaGetMatchesMapDeltaWithSelfEdge) {
    long vertex = 10;
    long current_block = B.block_assignment(vertex);
    EdgeWeights out_edges = finetune::edge_weights(graph.out_neighbors(vertex), vertex, false);
    EdgeWeights in_edges = finetune::edge_weights(graph.in_neighbors(vertex), vertex, false);
    args.coodelta = false;
    Delta map_delta = finetune::blockmodel_delta(vertex, current_block, Proposal.proposal, out_edges, in_edges, B);
    args.coodelta = true;
    Delta coo_delta = finetune::blockmodel_delta(vertex, current_block, Proposal.proposal, out_edges, in_edges, B);
    args.coodelta = false;
    EXPECT_EQ(coo_delta.self_edge_weight(), map_delta.self_edge_weight());
    // self_edge_weight should be 1 for vertex 10
    EXPECT_EQ(coo_delta.self_edge_weight(), 1);
    // All touched cells must agree
    for (const auto &e : map_delta.entries()) {
        long row = std::get<0>(e), col = std::get<1>(e);
        EXPECT_EQ(coo_delta.get(row, col), map_delta.get(row, col))
            << "Mismatch at (" << row << "," << col << ")";
    }
}

/// TODO: same test but using a vertex with a self edge
TEST_F(FinetuneTest, BlockmodelDeltasShouldSumUpToZero) {
    long vertex = 7;
    long current_block = B.block_assignment(vertex);
    EdgeWeights out_edges = finetune::edge_weights(graph.out_neighbors(vertex), vertex, false);
    EdgeWeights in_edges = finetune::edge_weights(graph.in_neighbors(vertex), vertex, false);
    Delta delta = finetune::blockmodel_delta(vertex, current_block, Proposal.proposal, out_edges, in_edges, B);
    long sum = 0;
    for (const auto &entry : delta.entries()) {
        sum += std::get<2>(entry);
    }
    EXPECT_EQ(sum, 0);
    vertex = 10;  // has a self-edge
    current_block = B.block_assignment(vertex);
    out_edges = finetune::edge_weights(graph.out_neighbors(vertex), vertex, false);
    in_edges = finetune::edge_weights(graph.in_neighbors(vertex), vertex, false);
    delta = finetune::blockmodel_delta(vertex, current_block, Proposal.proposal, out_edges, in_edges, B);
    sum = 0;
    for (const auto &entry : delta.entries()) {
        sum += std::get<2>(entry);
    }
    EXPECT_EQ(sum, 0);
}

TEST_F(FinetuneTest, BlockmodelDeltaGivesSameBlockmatrixAsEdgeCountUpdates) {
    long vertex = 7;
    long current_block = B.block_assignment(vertex);
    EdgeWeights out_edges = finetune::edge_weights(graph.out_neighbors(vertex), vertex);
    EdgeWeights in_edges = finetune::edge_weights(graph.in_neighbors(vertex), vertex);
    B.print_blockmatrix();
    Blockmodel B1 = B.copy();
    B1.move_vertex(V7, current_block, Proposal.proposal, Updates, new_block_degrees.block_degrees_out,
                  new_block_degrees.block_degrees_in, new_block_degrees.block_degrees);
    B1.print_blockmatrix();
    Blockmodel B2 = B.copy();
    B2.move_vertex(V7, Proposal.proposal, Deltas, new_block_degrees.block_degrees_out,
                   new_block_degrees.block_degrees_in, new_block_degrees.block_degrees);
    B2.print_blockmatrix();
    for (long row = 0; row < B.num_blocks(); ++row) {
        for (long col = 0; col < B.num_blocks(); ++col) {
            long val1 = B1.blockmatrix()->get(row, col);
            long val2 = B2.blockmatrix()->get(row, col);
            EXPECT_EQ(val1, val2)
                << "Blockmatrices differ at " << row << "," << col << " : using updates, value = " << val1
                << " using deltas, value = " << val2;
        }
    }
}

TEST_F(FinetuneTest, HastingsCorrectionBlockCountsAreTheSameWithAndWithoutBlockmodelDeltas) {
    long vertex = 7;
    MapVector<long> block_counts1;
//    std::unordered_map<long, long> block_counts1;
    for (const long neighbor : graph.out_neighbors(vertex)) {
        long neighbor_block = B.block_assignment(neighbor);
        block_counts1[neighbor_block] += 1;
    }
    for (const long neighbor : graph.in_neighbors(vertex)) {
        if (neighbor == vertex) continue;
        long neighbor_block = B.block_assignment(neighbor);
        block_counts1[neighbor_block] += 1;
    }
    utils::print(block_counts1);
    EdgeWeights out_edges = finetune::edge_weights(graph.out_neighbors(vertex), vertex);
    EdgeWeights in_edges = finetune::edge_weights(graph.in_neighbors(vertex), vertex);
    EdgeWeights blocks_out_neighbors = finetune::block_edge_weights(B.block_assignment(), out_edges);
    EdgeWeights blocks_in_neighbors = finetune::block_edge_weights(B.block_assignment(), in_edges);
    MapVector<long> block_counts2;
//    std::unordered_map<long, long> block_counts2;
    for (ulong i = 0; i < blocks_out_neighbors.indices.size(); ++i) {
        long block = blocks_out_neighbors.indices[i];
        long weight = blocks_out_neighbors.values[i];
        block_counts2[block] += weight; // block_count[new block] should initialize to 0
    }
    for (ulong i = 0; i < blocks_in_neighbors.indices.size(); ++i) {
        long block = blocks_in_neighbors.indices[i];
        long weight = blocks_in_neighbors.values[i];
        block_counts2[block] += weight; // block_count[new block] should initialize to 0
    }
    utils::print(block_counts2);
    for (const auto entry : block_counts1) {
        EXPECT_EQ(entry.second, block_counts2[entry.first]);
    }
    for (const auto entry : block_counts2) {
        EXPECT_EQ(entry.second, block_counts1[entry.first]);
    }
}

TEST_F(FinetuneTest, SpecialCaseGivesCorrectSparseEdgeCountUpdates) {
    long vertex = 6;
    long current_block = B3.block_assignment(vertex);
    EdgeWeights out_edges = finetune::edge_weights(graph.out_neighbors(vertex), vertex, false);
    EdgeWeights in_edges = finetune::edge_weights(graph.in_neighbors(vertex), vertex, true);
    SparseEdgeCountUpdates updates;
    finetune::edge_count_updates_sparse(B3, vertex, current_block, 0, out_edges, in_edges, updates);
    EXPECT_EQ(updates.block_row[0], 1);
    EXPECT_EQ(updates.block_row[1], 0);
    EXPECT_EQ(updates.block_row[2], 1);
    EXPECT_EQ(updates.block_row[3], 1);
    EXPECT_EQ(updates.block_row[4], 1);
    EXPECT_EQ(updates.block_row[5], 0);
    EXPECT_EQ(updates.block_col[0], 0);
    EXPECT_EQ(updates.block_col[1], 1);
    EXPECT_EQ(updates.block_col[2], 0);
    EXPECT_EQ(updates.block_col[3], 1);
    EXPECT_EQ(updates.block_col[4], 0);
    EXPECT_EQ(updates.block_col[5], 1);
    EXPECT_EQ(updates.proposal_row[0], 4);
    EXPECT_EQ(updates.proposal_row[1], 1);
    EXPECT_EQ(updates.proposal_row[2], 1);
    EXPECT_EQ(updates.proposal_row[3], 0);
    EXPECT_EQ(updates.proposal_row[4], 0);
    EXPECT_EQ(updates.proposal_row[5], 0);
    EXPECT_EQ(updates.proposal_col[0], 4);
    EXPECT_EQ(updates.proposal_col[1], 2);
    EXPECT_EQ(updates.proposal_col[2], 2);
    EXPECT_EQ(updates.proposal_col[3], 1);
    EXPECT_EQ(updates.proposal_col[4], 0);
    EXPECT_EQ(updates.proposal_col[5], 0);
}

TEST_F(FinetuneTest, SpecialCaseBlockmodelDeltasAreCorrect) {
    long vertex = 6;
    utils::ProposalAndEdgeCounts proposal {0, 1, 2, 3 };
    EdgeWeights out_edges = finetune::edge_weights(graph.out_neighbors(vertex), vertex, false);
    EdgeWeights in_edges = finetune::edge_weights(graph.in_neighbors(vertex), vertex, true);
    Delta delta = finetune::blockmodel_delta(vertex, 3, proposal.proposal, out_edges, in_edges, B3);
    EXPECT_EQ(delta.entries().size(), 6);
    EXPECT_EQ(delta.get(0,2), 1);
    EXPECT_EQ(delta.get(2,0), 1);
    EXPECT_EQ(delta.get(2,3), -1);
    EXPECT_EQ(delta.get(3,0), 1);
    EXPECT_EQ(delta.get(3,2), -1);
    EXPECT_EQ(delta.get(3,3), -1);
}

// Dense matrix versions

TEST_F(FinetuneDenseTest, SparseEdgeCountUpdatesAreCorrect) {
    long vertex = 7;
    long current_block = B.block_assignment(vertex);
    EdgeWeights out_edges = finetune::edge_weights(graph.out_neighbors(vertex), vertex, false);
    EdgeWeights in_edges = finetune::edge_weights(graph.in_neighbors(vertex), vertex, true);
    EdgeWeights blocks_out_neighbors = finetune::block_edge_weights(B.block_assignment(), out_edges);
    EdgeWeights blocks_in_neighbors = finetune::block_edge_weights(B.block_assignment(), in_edges);
    SparseEdgeCountUpdates updates;
    finetune::edge_count_updates_sparse(B, 7, current_block, 0, out_edges, in_edges, updates);
    EXPECT_EQ(updates.block_row[0], 2);
    EXPECT_EQ(updates.block_row[1], 1);
    EXPECT_EQ(updates.block_row[2], 3);
    EXPECT_EQ(updates.block_col[0], 1);
    EXPECT_EQ(updates.block_col[2], 3);
    EXPECT_EQ(updates.proposal_row[0], 8);
    EXPECT_EQ(updates.proposal_row[1], 1);
    EXPECT_EQ(updates.proposal_row[2], 1);
    EXPECT_EQ(updates.proposal_col[0], 8);
    EXPECT_EQ(updates.proposal_col[1], 2);
    EXPECT_EQ(updates.proposal_col[2], 2);
}

TEST_F(FinetuneDenseTest, SparseEdgeCountUpdatesWithSelfEdgesAreCorrect) {
    long vertex = 5;
    long current_block = B.block_assignment(vertex);
    EdgeWeights out_edges = finetune::edge_weights(graph.out_neighbors(vertex), vertex, false);
    EdgeWeights in_edges = finetune::edge_weights(graph.in_neighbors(vertex), vertex, true);
    EdgeWeights blocks_out_neighbors = finetune::block_edge_weights(B.block_assignment(), out_edges);
    EdgeWeights blocks_in_neighbors = finetune::block_edge_weights(B.block_assignment(), in_edges);
    SparseEdgeCountUpdates updates;
    finetune::edge_count_updates_sparse(B, 5, current_block, 0, out_edges, in_edges, updates);
    EXPECT_EQ(updates.block_row[0], 1);
    EXPECT_EQ(updates.block_row[1], 2);
    EXPECT_EQ(updates.block_col[0], 2);
    EXPECT_EQ(updates.block_col[1], 2);
    EXPECT_EQ(updates.proposal_row[0], 9);
    EXPECT_EQ(updates.proposal_row[1], 2);
    EXPECT_EQ(updates.proposal_row[2], 1);
    EXPECT_EQ(updates.proposal_col[0], 9);
    EXPECT_EQ(updates.proposal_col[1], 1);
    EXPECT_EQ(updates.proposal_col[2], 2);
}

TEST_F(FinetuneDenseTest, BlockmodelDeltasAreCorrect) {
    long vertex = 7;
    long current_block = B.block_assignment(vertex);
    EdgeWeights out_edges = finetune::edge_weights(graph.out_neighbors(vertex), vertex, false);
    EdgeWeights in_edges = finetune::edge_weights(graph.in_neighbors(vertex), vertex, false);
    Delta delta = finetune::blockmodel_delta(vertex, current_block, Proposal.proposal, out_edges, in_edges, B);
    EXPECT_EQ(delta.entries().size(), 6) << "blockmodel deltas are the wrong size. Expected 6 but got " << delta.entries().size();
    EXPECT_EQ(delta.get(0,0), 1);
    EXPECT_EQ(delta.get(0,1), 0);
    EXPECT_EQ(delta.get(0,2), 1);
    EXPECT_EQ(delta.get(1,0), 1);
    EXPECT_EQ(delta.get(1,2), -1);
    EXPECT_EQ(delta.get(2,0), 1);
    EXPECT_EQ(delta.get(2,1), 0);
    EXPECT_EQ(delta.get(2,2), -3);
}

TEST_F(FinetuneDenseTest, BlockmodelDeltasShouldSumUpToZero) {
    long vertex = 7;
    long current_block = B.block_assignment(vertex);
    EdgeWeights out_edges = finetune::edge_weights(graph.out_neighbors(vertex), vertex, false);
    EdgeWeights in_edges = finetune::edge_weights(graph.in_neighbors(vertex), vertex, false);
    Delta delta = finetune::blockmodel_delta(vertex, current_block, Proposal.proposal, out_edges, in_edges, B);
    long sum = 0;
    for (const auto &entry : delta.entries()) {
        sum += std::get<2>(entry);
    }
    EXPECT_EQ(sum, 0);
    vertex = 10;  // has a self-edge
    current_block = B.block_assignment(vertex);
    out_edges = finetune::edge_weights(graph.out_neighbors(vertex), vertex, false);
    in_edges = finetune::edge_weights(graph.in_neighbors(vertex), vertex, false);
    delta = finetune::blockmodel_delta(vertex, current_block, Proposal.proposal, out_edges, in_edges, B);
    sum = 0;
    for (const auto &entry : delta.entries()) {
        sum += std::get<2>(entry);
    }
    EXPECT_EQ(sum, 0);
}

TEST_F(FinetuneDenseTest, BlockmodelDeltaGivesSameBlockmatrixAsEdgeCountUpdates) {
    long vertex = 7;
    long current_block = B.block_assignment(vertex);
    EdgeWeights out_edges = finetune::edge_weights(graph.out_neighbors(vertex), vertex);
    EdgeWeights in_edges = finetune::edge_weights(graph.in_neighbors(vertex), vertex);
    B.print_blockmatrix();
    Blockmodel B1 = B.copy();
    B1.move_vertex(V7, current_block, Proposal.proposal, Updates, new_block_degrees.block_degrees_out,
                  new_block_degrees.block_degrees_in, new_block_degrees.block_degrees);
    B1.print_blockmatrix();
    Blockmodel B2 = B.copy();
    B2.move_vertex(V7, Proposal.proposal, Deltas, new_block_degrees.block_degrees_out,
                   new_block_degrees.block_degrees_in, new_block_degrees.block_degrees);
    B2.print_blockmatrix();
    for (long row = 0; row < B.num_blocks(); ++row) {
        for (long col = 0; col < B.num_blocks(); ++col) {
            long val1 = B1.blockmatrix()->get(row, col);
            long val2 = B2.blockmatrix()->get(row, col);
            EXPECT_EQ(val1, val2)
                << "Blockmatrices differ at " << row << "," << col << " : using updates, value = " << val1
                << " using deltas, value = " << val2;
        }
    }
}

TEST_F(FinetuneDenseTest, HastingsCorrectionBlockCountsAreTheSameWithAndWithoutBlockmodelDeltas) {
    long vertex = 7;
    MapVector<long> block_counts1;
    for (const long neighbor : graph.out_neighbors(vertex)) {
        long neighbor_block = B.block_assignment(neighbor);
        block_counts1[neighbor_block] += 1;
    }
    for (const long neighbor : graph.in_neighbors(vertex)) {
        if (neighbor == vertex) continue;
        long neighbor_block = B.block_assignment(neighbor);
        block_counts1[neighbor_block] += 1;
    }
    utils::print(block_counts1);
    EdgeWeights out_edges = finetune::edge_weights(graph.out_neighbors(vertex), vertex);
    EdgeWeights in_edges = finetune::edge_weights(graph.in_neighbors(vertex), vertex);
    EdgeWeights blocks_out_neighbors = finetune::block_edge_weights(B.block_assignment(), out_edges);
    EdgeWeights blocks_in_neighbors = finetune::block_edge_weights(B.block_assignment(), in_edges);
    MapVector<long> block_counts2;
    for (ulong i = 0; i < blocks_out_neighbors.indices.size(); ++i) {
        long block = blocks_out_neighbors.indices[i];
        long weight = blocks_out_neighbors.values[i];
        block_counts2[block] += weight;
    }
    for (ulong i = 0; i < blocks_in_neighbors.indices.size(); ++i) {
        long block = blocks_in_neighbors.indices[i];
        long weight = blocks_in_neighbors.values[i];
        block_counts2[block] += weight;
    }
    utils::print(block_counts2);
    for (const auto entry : block_counts1) {
        EXPECT_EQ(entry.second, block_counts2[entry.first]);
    }
    for (const auto entry : block_counts2) {
        EXPECT_EQ(entry.second, block_counts1[entry.first]);
    }
}

TEST_F(FinetuneDenseTest, SpecialCaseGivesCorrectSparseEdgeCountUpdates) {
    long vertex = 6;
    long current_block = B3.block_assignment(vertex);
    EdgeWeights out_edges = finetune::edge_weights(graph.out_neighbors(vertex), vertex, false);
    EdgeWeights in_edges = finetune::edge_weights(graph.in_neighbors(vertex), vertex, true);
    SparseEdgeCountUpdates updates;
    finetune::edge_count_updates_sparse(B3, vertex, current_block, 0, out_edges, in_edges, updates);
    EXPECT_EQ(updates.block_row[0], 1);
    EXPECT_EQ(updates.block_row[1], 0);
    EXPECT_EQ(updates.block_row[2], 1);
    EXPECT_EQ(updates.block_row[3], 1);
    EXPECT_EQ(updates.block_row[4], 1);
    EXPECT_EQ(updates.block_row[5], 0);
    EXPECT_EQ(updates.block_col[0], 0);
    EXPECT_EQ(updates.block_col[1], 1);
    EXPECT_EQ(updates.block_col[2], 0);
    EXPECT_EQ(updates.block_col[3], 1);
    EXPECT_EQ(updates.block_col[4], 0);
    EXPECT_EQ(updates.block_col[5], 1);
    EXPECT_EQ(updates.proposal_row[0], 4);
    EXPECT_EQ(updates.proposal_row[1], 1);
    EXPECT_EQ(updates.proposal_row[2], 1);
    EXPECT_EQ(updates.proposal_row[3], 0);
    EXPECT_EQ(updates.proposal_row[4], 0);
    EXPECT_EQ(updates.proposal_row[5], 0);
    EXPECT_EQ(updates.proposal_col[0], 4);
    EXPECT_EQ(updates.proposal_col[1], 2);
    EXPECT_EQ(updates.proposal_col[2], 2);
    EXPECT_EQ(updates.proposal_col[3], 1);
    EXPECT_EQ(updates.proposal_col[4], 0);
    EXPECT_EQ(updates.proposal_col[5], 0);
}

TEST_F(FinetuneDenseTest, SpecialCaseBlockmodelDeltasAreCorrect) {
    long vertex = 6;
    utils::ProposalAndEdgeCounts proposal {0, 1, 2, 3 };
    EdgeWeights out_edges = finetune::edge_weights(graph.out_neighbors(vertex), vertex, false);
    EdgeWeights in_edges = finetune::edge_weights(graph.in_neighbors(vertex), vertex, true);
    Delta delta = finetune::blockmodel_delta(vertex, 3, proposal.proposal, out_edges, in_edges, B3);
    EXPECT_EQ(delta.entries().size(), 6);
    EXPECT_EQ(delta.get(0,2), 1);
    EXPECT_EQ(delta.get(2,0), 1);
    EXPECT_EQ(delta.get(2,3), -1);
    EXPECT_EQ(delta.get(3,0), 1);
    EXPECT_EQ(delta.get(3,2), -1);
    EXPECT_EQ(delta.get(3,3), -1);
}

// COO-mode variants: build a COO delta from the same add() calls as ToySetUp's Deltas and apply it
// through move_vertex; the resulting blockmatrix must match the edge-count-update path (B1).

static Delta make_coo_finetune_deltas() {
    Delta d(2, 0, 10, true);
    d.add(0, 0, 1);
    d.add(0, 2, 1);
    d.add(1, 0, 1);
    d.add(1, 2, -1);
    d.add(2, 0, 1);
    d.add(2, 2, -3);
    return d;
}

TEST_F(FinetuneTest, CooBlockmodelDeltaGivesSameBlockmatrixAsEdgeCountUpdates) {
    long vertex = 7;
    long current_block = B.block_assignment(vertex);
    Blockmodel B1 = B.copy();
    B1.move_vertex(V7, current_block, Proposal.proposal, Updates, new_block_degrees.block_degrees_out,
                  new_block_degrees.block_degrees_in, new_block_degrees.block_degrees);
    Delta coo_deltas = make_coo_finetune_deltas();
    Blockmodel B2 = B.copy();
    B2.move_vertex(V7, Proposal.proposal, coo_deltas, new_block_degrees.block_degrees_out,
                   new_block_degrees.block_degrees_in, new_block_degrees.block_degrees);
    for (long row = 0; row < B.num_blocks(); ++row) {
        for (long col = 0; col < B.num_blocks(); ++col) {
            long val1 = B1.blockmatrix()->get(row, col);
            long val2 = B2.blockmatrix()->get(row, col);
            EXPECT_EQ(val1, val2)
                << "Blockmatrices differ at " << row << "," << col << " : using updates, value = " << val1
                << " using COO deltas, value = " << val2;
        }
    }
}

TEST_F(FinetuneDenseTest, CooBlockmodelDeltaGivesSameBlockmatrixAsEdgeCountUpdates) {
    long vertex = 7;
    long current_block = B.block_assignment(vertex);
    Blockmodel B1 = B.copy();
    B1.move_vertex(V7, current_block, Proposal.proposal, Updates, new_block_degrees.block_degrees_out,
                  new_block_degrees.block_degrees_in, new_block_degrees.block_degrees);
    Delta coo_deltas = make_coo_finetune_deltas();
    Blockmodel B2 = B.copy();
    B2.move_vertex(V7, Proposal.proposal, coo_deltas, new_block_degrees.block_degrees_out,
                   new_block_degrees.block_degrees_in, new_block_degrees.block_degrees);
    for (long row = 0; row < B.num_blocks(); ++row) {
        for (long col = 0; col < B.num_blocks(); ++col) {
            long val1 = B1.blockmatrix()->get(row, col);
            long val2 = B2.blockmatrix()->get(row, col);
            EXPECT_EQ(val1, val2)
                << "Blockmatrices differ at " << row << "," << col << " : using updates, value = " << val1
                << " using COO deltas, value = " << val2;
        }
    }
}
