#include <vector>

#include <gtest/gtest.h>

#include "entropy.hpp"
#include "block_merge.hpp"
#include "blockmodel/delta.hpp"

#include "toy_example.hpp"

class BlockMergeDenseTest : public BlockMergeTest {
    void SetUp() override {
        forced_matrix_type = "dense";
        BlockMergeTest::SetUp();
    }
};

TEST_F(BlockMergeTest, BlockmodelDeltaIsCorrectlyComputed) {
    Delta delta = block_merge::blockmodel_delta(0, 1, B);
    EXPECT_EQ(delta.entries().size(), 6);
    EXPECT_EQ(delta.get(0,0), -7);
    EXPECT_EQ(delta.get(0,1), -1);
    EXPECT_EQ(delta.get(1,0), -1);
    EXPECT_EQ(delta.get(1,1), 9);
    EXPECT_EQ(delta.get(1,2), 0);
    EXPECT_EQ(delta.get(2,0), -1);
    EXPECT_EQ(delta.get(2,1), 1);
}

TEST_F(BlockMergeDenseTest, BlockmodelDeltaIsCorrectlyComputed) {
    Delta delta = block_merge::blockmodel_delta(0, 1, B);
    EXPECT_EQ(delta.entries().size(), 6);
    EXPECT_EQ(delta.get(0,0), -7);
    EXPECT_EQ(delta.get(0,1), -1);
    EXPECT_EQ(delta.get(1,0), -1);
    EXPECT_EQ(delta.get(1,1), 9);
    EXPECT_EQ(delta.get(1,2), 0);
    EXPECT_EQ(delta.get(2,0), -1);
    EXPECT_EQ(delta.get(2,1), 1);
}

// COO-mode variants: same values expected, zero entry (1,2) absent from COO storage but get() still returns 0.

TEST_F(BlockMergeTest, CooBlockmodelDeltaIsCorrectlyComputed) {
    args.coodelta = true;
    Delta delta = block_merge::blockmodel_delta(0, 1, B);
    args.coodelta = false;
    EXPECT_EQ(delta.get(0,0), -7);
    EXPECT_EQ(delta.get(0,1), -1);
    EXPECT_EQ(delta.get(1,0), -1);
    EXPECT_EQ(delta.get(1,1), 9);
    EXPECT_EQ(delta.get(1,2), 0);
    EXPECT_EQ(delta.get(2,0), -1);
    EXPECT_EQ(delta.get(2,1), 1);
}

TEST_F(BlockMergeDenseTest, CooBlockmodelDeltaIsCorrectlyComputed) {
    args.coodelta = true;
    Delta delta = block_merge::blockmodel_delta(0, 1, B);
    args.coodelta = false;
    EXPECT_EQ(delta.get(0,0), -7);
    EXPECT_EQ(delta.get(0,1), -1);
    EXPECT_EQ(delta.get(1,0), -1);
    EXPECT_EQ(delta.get(1,1), 9);
    EXPECT_EQ(delta.get(1,2), 0);
    EXPECT_EQ(delta.get(2,0), -1);
    EXPECT_EQ(delta.get(2,1), 1);
}

// =============================================================================
// Dense-vs-sparse equivalence tests (block merge path)
// =============================================================================

class BlockMergeDenseSparseEquivTest : public ::testing::Test {
protected:
    Graph graph;
    Blockmodel B_sparse, B_dense;
    Delta Deltas;
    common::NewBlockDegrees block_degrees;

    void SetUp() override {
        args.parametric = true;
        std::vector<std::vector<long>> edges {
            {0,0},{0,1},{0,2},{1,2},{2,3},{3,1},{3,2},{3,5},{4,1},{4,6},{5,4},{5,5},{5,6},{5,7},
            {6,4},{7,3},{7,9},{8,5},{8,7},{9,10},{10,7},{10,8},{10,10}
        };
        std::vector<long> assignment = { 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2 };
        std::vector<bool> self_edges = { true, false, false, false, false, true, false, false, false, false, true };
        NeighborList out_n, in_n;
        for (const auto &e : edges) {
            utils::insert(out_n, e[0], e[1]);
            utils::insert(in_n, e[1], e[0]);
        }
        graph = Graph(out_n, in_n, 11, (long) edges.size(), self_edges, assignment);
        // BlockMergeTest setup: transpose=true, merging block 0 into block 1
        args.matrix_type = "sparse";
        B_sparse = Blockmodel(3, graph, 0.5, assignment);
        args.matrix_type = "dense";
        B_dense = Blockmodel(3, graph, 0.5, assignment);
        args.matrix_type = "sparse";
        Deltas = Delta(0, 1);
        Deltas.add(0, 0, -7); Deltas.add(0, 1, -1); Deltas.add(1, 0, -1);
        Deltas.add(1, 1, 9); Deltas.add(2, 0, -1); Deltas.add(2, 1, 1);
        block_degrees.block_degrees_out = { 0, 15, 8 };
        block_degrees.block_degrees_in  = { 0, 16, 7 };
        block_degrees.block_degrees     = { 0, 17, 9 };
    }
};

TEST_F(BlockMergeDenseSparseEquivTest, BlockmodelDeltaEntriesMatchBetweenDenseAndSparse) {
    args.matrix_type = "sparse";
    Delta sparse_delta = block_merge::blockmodel_delta(0, 1, B_sparse);
    args.matrix_type = "dense";
    Delta dense_delta = block_merge::blockmodel_delta(0, 1, B_dense);
    args.matrix_type = "sparse";
    EXPECT_EQ(sparse_delta.entries().size(), dense_delta.entries().size());
    for (const auto &entry : sparse_delta.entries()) {
        long r = std::get<0>(entry), c = std::get<1>(entry), v = std::get<2>(entry);
        EXPECT_EQ(v, dense_delta.get(r, c)) << "delta(" << r << "," << c << ")";
    }
}

TEST_F(BlockMergeDenseSparseEquivTest, BlockMergeDeltaMDLMatchesBetweenDenseAndSparse) {
    args.matrix_type = "sparse";
    double sparse_dE = entropy::block_merge_delta_mdl(0, B_sparse, Deltas, block_degrees);
    args.matrix_type = "dense";
    double dense_dE = entropy::block_merge_delta_mdl(0, B_dense, Deltas, block_degrees);
    args.matrix_type = "sparse";
    EXPECT_FLOAT_EQ(sparse_dE, dense_dE);
}

TEST_F(BlockMergeDenseSparseEquivTest, NonparametricBlockMergeDeltaMDLMatchesBetweenDenseAndSparse) {
    args.parametric = false;
    utils::ProposalAndEdgeCounts proposal = { 1, 17, 16, 23 };
    args.matrix_type = "sparse";
    double sparse_dE = entropy::nonparametric::block_merge_delta_mdl(B_sparse, proposal, graph, Deltas);
    args.matrix_type = "dense";
    double dense_dE = entropy::nonparametric::block_merge_delta_mdl(B_dense, proposal, graph, Deltas);
    args.matrix_type = "sparse";
    args.parametric = true;
    EXPECT_FLOAT_EQ(sparse_dE, dense_dE);
}
