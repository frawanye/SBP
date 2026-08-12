#include "finetune.hpp"

#include "delta_coo.hpp"
#include "entropy.hpp"
#include <omp.h>
#include "pcg_random.hpp"

// #pragma omp requires unified_shared_memory 

namespace finetune {

namespace gpu {

Blockmodel &asynchronous_gibbs(Blockmodel &blockmodel, const Graph &graph, bool golden_ratio_not_reached) {
    std::cout << "Asynchronous Gibbs iteration" << std::endl;
    if (blockmodel.num_blocks() == 1) {
        return blockmodel;
    }
    std::vector<double> delta_entropies;
    std::vector<long> vertex_moves;
    std::vector<long> vertices = utils::range<long>(0, graph.num_vertices());
    long total_vertex_moves = 0;
    blockmodel.setOverall_entropy(entropy::mdl(blockmodel, graph));
    double initial_entropy = blockmodel.getOverall_entropy();
    double last_entropy = initial_entropy;
    std::vector<long> shuffled_vertices = utils::range<long>(0, graph.num_vertices());  // TODO: make this not a std::vector
    for (long iteration = 0; iteration < MAX_NUM_ITERATIONS; ++iteration) {
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        if (!args.ordered) {
            std::shuffle(shuffled_vertices.begin(), shuffled_vertices.end(), std::mt19937_64(seed));
        }
        long* shuffled_vertices_gpu = shuffled_vertices.data();
        long _vertex_moves = 0;
        double num_batches = args.batches;
        long batch_size = long(ceil(double(graph.num_vertices()) / num_batches));
        for (long batch = 0; batch < graph.num_vertices() / batch_size; ++batch) {
            long start = batch * batch_size;
            long end = std::min(graph.num_vertices(), (batch + 1) * batch_size);
            // Block assignment used to re-create the Blockmodel after each batch to improve mixing time of
            // asynchronous Gibbs sampling
            std::vector<VertexMoveGPU> moves(graph.num_vertices());
            VertexMoveGPU* moves_gpu = moves.data();
            BlockmodelGPUView blockmodel_gpu = blockmodel.gpu_view();
            double start_t = MPI_Wtime();
            const CSR &graph_csr = graph.out_csr();
            const CSR &graph_csc = graph.in_csr();
            if (args.sparse_entries_ds) {
                // Refresh the neighbor-block companions: assignments change between batches, and
                // the sort scrambles each slice relative to col_indices.
                #pragma omp target teams distribute parallel for
                for (long vertex = 0; vertex < graph_csr.nrows; ++vertex) {
                    refresh_neighbor_blocks(vertex, graph_csr, graph_csc, blockmodel_gpu);
                }
                // entries_dS is serial per vertex on this path, so the team's threads are handed
                // one vertex each rather than cooperating on a block loop.
                #pragma omp target teams distribute parallel for thread_limit(args.gpu_thread_limit)
                for (long index = start; index < end; ++index) {
                    long vertex = shuffled_vertices_gpu[index];
                    pcg32 rng(seed + (uint64_t)iteration, (uint64_t)vertex);
                    VertexMoveGPU proposal = propose_gibbs_move(blockmodel_gpu, vertex, graph_csr, graph_csc, rng);
                    moves_gpu[vertex] = proposal;
                }
            } else {
                // entries_dS opens its own parallel region over the B blocks, so one thread per
                // team enters the loop here.
                #pragma omp target teams distribute thread_limit(args.gpu_thread_limit)
                for (long index = start; index < end; ++index) {
                    long vertex = shuffled_vertices_gpu[index];
                    pcg32 rng(seed + (uint64_t)iteration, (uint64_t)vertex);
                    VertexMoveGPU proposal = propose_gibbs_move(blockmodel_gpu, vertex, graph_csr, graph_csc, rng);
                    moves_gpu[vertex] = proposal;
                }
            }
            double parallel_t = MPI_Wtime();
            timers::MCMC_parallel_time += parallel_t - start_t;
            for (const VertexMoveGPU &move : moves) {
                if (!move.did_move) continue;
                EdgeWeights out_edges = edge_weights(graph.out_neighbors(move.vertex.id), move.vertex.id, false);
                EdgeWeights in_edges = edge_weights(graph.in_neighbors(move.vertex.id), move.vertex.id, true);
                if (blockmodel.move_vertex_gpu(move, out_edges, in_edges)) {
                    _vertex_moves++;
                }
            }
            timers::MCMC_vertex_move_time += MPI_Wtime() - parallel_t;
        }
        double entropy = entropy::mdl(blockmodel, graph);
        double delta_entropy = entropy - last_entropy;
        delta_entropies.push_back(delta_entropy);
        last_entropy = entropy;
        vertex_moves.push_back(_vertex_moves);
        timers::MCMC_moves += _vertex_moves;
        std::cout << "Itr: " << iteration << ", number of vertex moves: " << _vertex_moves << ", delta S: ";
        std::cout << delta_entropy / initial_entropy << std::endl;
        total_vertex_moves += _vertex_moves;
        timers::MCMC_iterations++;
        // Early stopping
        if (early_stop(iteration, golden_ratio_not_reached, initial_entropy, delta_entropies)) {
            break;
        }
    }
    blockmodel.setOverall_entropy(entropy::mdl(blockmodel, graph));
    std::cout << "Total number of vertex moves: " << total_vertex_moves << ", overall entropy: ";
    std::cout << blockmodel.getOverall_entropy() << std::endl;
    return blockmodel;
}

#pragma omp declare target
//std::ofstream my_file;

DeltaCOO blockmodel_delta(long vertex, long current_block, long proposed_block, const CSR &graph_csr, 
                          const CSR &graph_csc, const BlockmodelGPUView &blockmodel) {
    DeltaCOO delta(current_block, proposed_block, 2*(graph_csr.neighbors(vertex).size() + graph_csc.neighbors(vertex).size()));

    // current_block -> current_block == proposed_block --> proposed_block  (this includes self edges)
    // current_block --> other_block == proposed_block --> other_block
    // other_block --> current_block == other_block --> proposed_block
    // current_block --> proposed_block == proposed_block --> proposed_block
    // proposed_block --> current_block == proposed_block --> proposed_block
    NeighborView out_neighbors = graph_csr.neighbors(vertex);
    NeighborView in_neighbors = graph_csc.neighbors(vertex);
    for (size_t i = 0; i < out_neighbors.size(); ++i) {
        long out_vertex = out_neighbors[i];
        long out_block = blockmodel.block_assignment(out_vertex);
        long edge_weight = out_neighbors.val(i);
        if (vertex == out_vertex) {
            delta.add(proposed_block, proposed_block, edge_weight);
            delta.self_edge_weight(1);
        } else {
            delta.add(proposed_block, out_block, edge_weight);
        }
        delta.sub(current_block, out_block, edge_weight);
    }
    for (size_t i = 0; i < in_neighbors.size(); ++i) {
        long in_vertex = in_neighbors[i];
        long in_block = blockmodel.block_assignment(in_vertex);
        long edge_weight = in_neighbors.val(i);
        if (vertex != in_vertex) {
            delta.add(in_block, proposed_block, edge_weight);
            delta.sub(in_block, current_block, edge_weight);
        }
    }
    return delta;
}

void heap_sort(long *keys, long *values, long size) {
    // First, we build the heap
    // The heap is stored in the array in level-order, root, left child, right child, left left, left right, right left,
    // right right, etc.
    // Left child of index is is 2*index + 1 and right child is 2*index + 2.
    // The parent of index is (index - 1) / 2.
    for (long start = size / 2 - 1; start >= 0; --start) { // start is the last parent node
        long root = start;
        long child, left_child, right_child;
        child = left_child = 2 * root + 1;
        right_child = left_child + 1; // 2 * root + 2
        if (left_child >= size)
            continue; // no children
        if (right_child < size && keys[left_child] < keys[right_child])
            child = right_child;
        if (keys[root] >= keys[child])
            continue; // if root is greater than the child, then the heap property is satisfied
        // else, swap the root and the child
        long temp_key = keys[root];
        keys[root] = keys[child];
        keys[child] = temp_key;
        long temp_value = values[root];
        values[root] = values[child];
        values[child] = temp_value;
        root = child;
        while (true) {
            child = 2 * root + 1;
            right_child = child + 1;
            if (child >= size)
                break;
            if (right_child < size && keys[child] < keys[right_child])
                child = right_child;
            if (keys[root] >= keys[child])
                break;
            // else, swap the root and the child
            temp_key = keys[root];
            keys[root] = keys[child];
            keys[child] = temp_key;
            temp_value = values[root];
            values[root] = values[child];
            values[child] = temp_value;
            root = child;
        }
    }
    // Now, we "sort" the heap by repeatedly extracting the root and then replacing it with the last element.
    for (long end = size - 1; end > 0; --end) {
        long temp_key = keys[0];
        keys[0] = keys[end];
        keys[end] = temp_key;
        long temp_value = values[0];
        values[0] = values[end];
        values[end] = temp_value;
        // Now, rebalance the heap
        long root = 0;
        while (true) {
            long child = 2 * root + 1;
            long right_child = child + 1;
            if (child >= end)
                break;
            if (right_child < end && keys[child] < keys[right_child])
                child = right_child;
            if (keys[root] >= keys[child])
                break;
            // else, swap the root and the child
            temp_key = keys[root];
            keys[root] = keys[child];
            keys[child] = temp_key;
            temp_value = values[root];
            values[root] = values[child];
            values[child] = temp_value;
            root = child;
        }
    }
}

void refresh_neighbor_blocks(long vertex, const CSR &graph_csr, const CSR &graph_csc,
                             const BlockmodelGPUView &blockmodel) {
    long csr_start = graph_csr.row_ptrs[vertex];
    long csr_size = graph_csr.degree(vertex);
    for (long i = csr_start; i < csr_start + csr_size; ++i) {
        graph_csr._block_id[i] = blockmodel.block_assignment(graph_csr.col_indices[i]);
        graph_csr._edge_weight[i] = graph_csr.vals[i];
    }
    heap_sort(graph_csr._block_id + csr_start, graph_csr._edge_weight + csr_start, csr_size);

    long csc_start = graph_csc.row_ptrs[vertex];
    long csc_size = graph_csc.degree(vertex);
    for (long i = csc_start; i < csc_start + csc_size; ++i) {
        graph_csc._block_id[i] = blockmodel.block_assignment(graph_csc.col_indices[i]);
        graph_csc._edge_weight[i] = graph_csc.vals[i];
    }
    heap_sort(graph_csc._block_id + csc_start, graph_csc._edge_weight + csc_start, csc_size);
}

VertexMoveGPU propose_gibbs_move(const BlockmodelGPUView &blockmodel, long vertex, const CSR &graph_csr, const CSR &graph_csc, pcg32 &rng) {
    bool did_move = false;
    long current_block = blockmodel.block_assignment(vertex);

    // EdgeWeights out_edges = edge_weights(graph_csr.neighbors(vertex), vertex, false);
    // EdgeWeights in_edges = edge_weights(graph_csc.neighbors(vertex), vertex, true);

    ProposedMove proposal = common::gpu::propose_new_block(
        vertex, current_block, graph_csr, graph_csc, blockmodel, rng
    );
    if (proposal.proposed_block == current_block) {  // TODO: can we somehow get rid of this branch? Always eval_vertex_move somehow?
        return VertexMoveGPU{0.0, did_move, {-1, -1, -1 }, -1};
    }
    return eval_vertex_move(vertex, current_block, proposal, blockmodel, graph_csr, graph_csc);
}

VertexMoveGPU eval_vertex_move(long vertex, long current_block, const ProposedMove &proposal,
                               const BlockmodelGPUView &blockmodel, const CSR &graph_csr, const CSR &graph_csc) {
                                // TODO: things like size and such should be ulong/size_t, not long.
    Vertex v = { vertex, graph_csr.neighbors(vertex).size(), long(graph_csc.neighbors(vertex).size()) };
    // const DeltaCOO delta = blockmodel_delta(vertex, current_block, proposal.proposed_block, graph_csr, graph_csc, blockmodel);
    // double hastings = entropy::gpu::hastings_correction(vertex, graph_csr, graph_csc, blockmodel, delta, current_block, proposal);
    double hastings = 1.0; // for now, GPU code only supports nonparametric mode, where we don't need a hastings_correction.
    // double delta_entropy = !args.parametric ?
    //                         entropy::gpu::nonparametric::delta_mdl(blockmodel, graph, vertex, delta, proposal) :
    //                         entropy::delta_mdl(blockmodel, delta, proposal);
    double delta_entropy = entropy::gpu::nonparametric::delta_mdl(blockmodel, graph_csr, graph_csc, proposal);

    if (accept(delta_entropy, hastings))  // TODO: can we simply return VertexMoveGPU{..., accept(), ...}? Will that breka anything downstream?
        return VertexMoveGPU{delta_entropy, true, v, proposal.proposed_block};
    return VertexMoveGPU{delta_entropy, false, InvalidVertex, -1};
}
#pragma omp end declare target

} // namespace gpu

} // namespace finetune