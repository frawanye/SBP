#include "common.hpp"

#include "args.hpp"
#include "mpi_data.hpp"
#include "rng.hpp"

#include <cassert>
#include "utils.hpp"
#include "typedefs.hpp"

namespace common {

std::uniform_int_distribution<long> candidates;

long choose_neighbor(std::vector<long> &neighbor_indices, std::vector<long> &neighbor_weights) {
    std::discrete_distribution<long> distribution(neighbor_weights.begin(), neighbor_weights.end());
    long index = distribution(rng::generator());
    return neighbor_indices[index];
}

long choose_neighbor(const std::vector<long> &block_weights) {
    // Dense multinomial: block_weights is indexed directly by block id, so the drawn index IS the
    // chosen block. std::discrete_distribution normalizes the (unnormalized) weights internally.
    std::discrete_distribution<long> distribution(block_weights.begin(), block_weights.end());
    return distribution(rng::generator());
}

long choose_neighbor(const SparseVector<double> &multinomial_distribution) {
//    std::cout << "multinomial distribution: ";
//    utils::print<long>(multinomial_distribution.idx);
//    utils::print<double>(multinomial_distribution.data);
    // std::cout << "in choose_neighbor" << std::endl;
    // std::cout << "data.size = " << multinomial_distribution.data.size() << " idx.size = ";
    // std::cout << multinomial_distribution.idx.size() << std::endl;
    std::discrete_distribution<long> distribution(
        multinomial_distribution.data.begin(), multinomial_distribution.data.end());
    size_t index = distribution(rng::generator());
    if (index >= multinomial_distribution.idx.size()) { std::cout << "ERROR: index " << index << " dist size = " << multinomial_distribution.idx.size() << std::endl; }
//     std::cout << "index = " << index << " out of " << multinomial_distribution.idx.size() << " | " << multinomial_distribution.data.size() << std::endl;
    return multinomial_distribution.idx[index];
}

NewBlockDegrees compute_new_block_degrees(long current_block, const Blockmodel &blockmodel, long current_block_self_edges,
                                          long proposed_block_self_edges, utils::ProposalAndEdgeCounts &proposal) {
    std::vector<long> degrees_out(blockmodel.degrees_out());
    std::vector<long> degrees_in(blockmodel.degrees_in());
    std::vector<long> degrees_total(blockmodel.degrees());
    degrees_out[current_block] -= proposal.num_out_neighbor_edges;
    degrees_out[proposal.proposal] += proposal.num_out_neighbor_edges;
    degrees_in[current_block] -= proposal.num_in_neighbor_edges;
    degrees_in[proposal.proposal] += proposal.num_in_neighbor_edges;
    degrees_total[current_block] = degrees_out[current_block] + degrees_in[current_block] - current_block_self_edges;
    degrees_total[proposal.proposal] = degrees_out[proposal.proposal] + degrees_in[proposal.proposal]
            - proposed_block_self_edges;
    return NewBlockDegrees{degrees_out, degrees_in, degrees_total};
}

double delta_entropy_temp(std::vector<long> &row_or_col, std::vector<long> &block_degrees, long degree, long num_edges) {
    if (args.undirected)
        return undirected::delta_entropy_temp(row_or_col, block_degrees, degree, num_edges);
    return directed::delta_entropy_temp(row_or_col, block_degrees, degree);
}

double delta_entropy_temp(const MapVector<long> &row_or_col, const std::vector<long> &block_degrees, long degree,
                          long num_edges) {
    if (args.undirected)
        return undirected::delta_entropy_temp(row_or_col, block_degrees, degree, num_edges);
    return directed::delta_entropy_temp(row_or_col, block_degrees, degree);
}

double delta_entropy_temp(const MapVector<long> &row_or_col, const std::vector<long> &block_degrees, long degree,
                          long current_block, long proposal, long num_edges) {
    if (args.undirected)
        return undirected::delta_entropy_temp(row_or_col, block_degrees, degree, current_block, proposal, num_edges);
    return directed::delta_entropy_temp(row_or_col, block_degrees, degree, current_block, proposal);
}

std::vector<long> exclude_indices(const std::vector<long> &in, long index1, long index2) {
    std::vector<long> out = utils::constant<long>((long) in.size() - 1, 0);
    long count = 0;
    for (long i = 0; i < (long) in.size(); ++i) {
        if (i == index1 || i == index2) {
            continue;
        }
        out[count] = in[i];
        count++;
    }
    return out;
}

MapVector<long>& exclude_indices(MapVector<long> &in, long index1, long index2) {
    // MapVector<long> out(in);
    in.erase(index1);
    in.erase(index2);
    return in;
}

std::vector<long> index_nonzero(const std::vector<long> &values, std::vector<long> &indices) {
    std::vector<long> results;
    for (size_t i = 0; i < indices.size(); ++i) {
        long index = indices[i];
        if (index != 0) {
            long value = values[i];
            results.push_back(value);
        }
    }
    return results;
}

std::vector<long> index_nonzero(const std::vector<long> &values, MapVector<long> &indices_map) {
    // if (omp_get_thread_num() == 0)
    //     std::cout << "sparse version" << std::endl;
    std::vector<long> results;
    for (const std::pair<long, long> &element : indices_map) {
        if (element.second != 0) {
            // if (omp_get_thread_num() == 0)
            //     std::cout << "pushing " << values[element.first] << " because " << element.first << " = " << element.second << std::endl;
            results.push_back(values[element.first]);
        }
    }
    return results;
}

std::vector<long> nonzeros(std::vector<long> &in) {
    std::vector<long> values;
    for (size_t i = 0; i < in.size(); ++i) {
        long value = in[i];
        if (value != 0) {
            values.push_back(value);
        }
    }
    return values;
}

std::vector<long> nonzeros(MapVector<long> &in) {
    std::vector<long> values;
    for (const auto &element : in) {
        if (element.second != 0) {
            values.push_back(element.second);
        }
    }
    return values;
}

// TODO: get rid of block_assignment, just use blockmodel?
utils::ProposalAndEdgeCounts propose_new_block(long current_block, EdgeWeights &out_blocks, EdgeWeights &in_blocks,
                                               const std::vector<long> &block_assignment, const Blockmodel &blockmodel,
                                               bool block_merge) {
    std::vector<long> neighbor_indices = utils::concatenate<long>(out_blocks.indices, in_blocks.indices);
    std::vector<long> neighbor_weights = utils::concatenate<long>(out_blocks.values, in_blocks.values);
    long k_out = std::accumulate(out_blocks.values.begin(), out_blocks.values.end(), 0);
    long k_in = std::accumulate(in_blocks.values.begin(), in_blocks.values.end(), 0);
    long k = k_out + k_in;
    long num_blocks = blockmodel.num_blocks();

    // If the current block has no neighbors, propose merge with random block
    if (k == 0) {
        long proposal = propose_random_block(current_block, num_blocks);
        return utils::ProposalAndEdgeCounts{proposal, k_out, k_in, k};
    }
    long neighbor_block;
    if (block_merge)
        neighbor_block = choose_neighbor(neighbor_indices, neighbor_weights);
    else {
        long neighbor = choose_neighbor(neighbor_indices, neighbor_weights);
        neighbor_block = block_assignment[neighbor];
    }

    // With a probability inversely proportional to block degree, propose a random block merge
    /* if (rng::generate() <= (_num_blocks / ((double) blockmodel.degrees(neighbor_block) + _num_blocks))) {
        long proposal = propose_random_block(current_block, _num_blocks);
        return utils::ProposalAndEdgeCounts{proposal, k_out, k_in, k};
    } */

    // Dense compute path: build the neighbor-weight multinomial directly from dense getrow/getcol
    // vectors (indexed by block id) instead of materializing a MapVector via neighbors_weights.
    if (dense_compute()) {
        const std::shared_ptr<ISparseMatrix> matrix = blockmodel.blockmatrix();
        std::vector<long> row = matrix->getrow(neighbor_block);
        std::vector<long> col = matrix->getcol(neighbor_block);
        std::vector<long> dense_edges(num_blocks, 0);
        // Mirror DenseMatrix::neighbors_weights: outgoing edges from the row (self-edge included),
        // incoming edges from the column excluding the diagonal to avoid double counting.
        for (long i = 0; i < num_blocks; ++i) {
            dense_edges[i] = row[i] + (i != neighbor_block ? col[i] : 0);
        }
        if (block_merge) {  // Make sure proposal != current_block
            dense_edges[current_block] = 0;
        }
        long dense_total = utils::sum<long>(dense_edges);
        if (dense_total == 0) {  // Neighbor block has no usable neighbors, so propose a random block
            long proposal = propose_random_block(current_block, num_blocks);
            return utils::ProposalAndEdgeCounts{proposal, k_out, k_in, k};
        }
        long proposal = choose_neighbor(dense_edges);
        return utils::ProposalAndEdgeCounts{proposal, k_out, k_in, k};
    }

    // Build multinomial distribution
    double total_edges = 0.0;
    MapVector<long> edges = blockmodel.blockmatrix()->neighbors_weights(neighbor_block);
//    const std::shared_ptr<ISparseMatrix> matrix = blockmodel.blockmatrix();
//    const MapVector<long> &col = matrix->getcol_sparse(neighbor_block);
//    MapVector<long> edges = blockmodel.blockmatrix()->getrow_sparse(neighbor_block);
//    for (const auto &pair : col) {
//        edges[pair.first] += pair.second;
//    }
    if (block_merge) {  // Make sure proposal != current_block
        edges[current_block] = 0;
        total_edges = utils::sum<double, long>(edges);
        if (total_edges == 0.0) { // Neighbor block has no neighbors, so propose a random block
            long proposal = propose_random_block(current_block, num_blocks);
            return utils::ProposalAndEdgeCounts{proposal, k_out, k_in, k};
        }
    } else {
        total_edges = utils::sum<double, long>(edges);
    }
    if (edges.empty()) {
        std::cerr << "ERROR " << "ERROR: NO EDGES for neighbor_block = " << neighbor_block << "! k = " << blockmodel.degrees(neighbor_block) << " "
                  << blockmodel.degrees_out(neighbor_block) << " " << blockmodel.degrees_in(neighbor_block)
                  << std::endl;
        utils::print<long>(blockmodel.blockmatrix()->getrow_sparse(neighbor_block));
        utils::print<long>(blockmodel.blockmatrix()->getcol_sparse(neighbor_block));
    }
    // Propose a block based on the multinomial distribution of block neighbor edges
    SparseVector<double> multinomial_distribution;
    utils::div(edges, total_edges, multinomial_distribution);
    long proposal = choose_neighbor(multinomial_distribution);
    return utils::ProposalAndEdgeCounts{proposal, k_out, k_in, k};
}

namespace gpu {

#pragma omp declare target

long random_integer(long max, pcg32 &rng) {
    return (long)(((uint64_t)rng() * (uint64_t)max) >> 32);
}

long choose_neighbor(long vertex, long vertex_degree, const NeighborView &out_neighbors, const NeighborView &in_neighbors, pcg32 &rng) {
    // long r = random_integer(0, vertex_degree);
    long r = random_integer(vertex_degree, rng);
    long cumulative_weight = 0;
    long chosen = -1;
    for (size_t i = 0; i < out_neighbors.size(); ++i) {
        cumulative_weight += out_neighbors.val(i);
        bool first = (chosen < 0) & (cumulative_weight > r);
        chosen = first ? out_neighbors[i] : chosen;  // if chosen = -1, and cumulative_weight > r, then choose this neighbor
    }
    for (size_t i = 0; i < in_neighbors.size(); ++i) {
        cumulative_weight += in_neighbors.val(i) * (long)(in_neighbors[i] != vertex); // do not count self-edges
        bool first = (chosen < 0) & (cumulative_weight > r);
        chosen = first ? in_neighbors[i] : chosen;  // if chosen = -1, and cumulative_weight > r, then choose this neighbor
    }
    return chosen; // something went VERY wrong here
}

long choose_neighbor(long num_blocks, const long* block_weights, long total_weight, pcg32 &rng) {
    long r = random_integer(total_weight, rng);
    long cumulative_weight = 0;
    long chosen = -1;
    for (size_t block = 0; block < num_blocks; ++block) {
        cumulative_weight += block_weights[block];
        bool first = (chosen < 0) & (cumulative_weight > r);
        chosen = first ? block : chosen;  // if chosen = -1, and cumulative_weight > r, then choose this neighbor
    }
    return chosen; // something went VERY wrong here
}

long choose_neighbor(long neighbor_block,const BlockmodelGPUView &blockmodel, long total_weight, pcg32 &rng) {
    long r = random_integer(total_weight, rng);
    long cumulative_weight = 0;
    long chosen = -1;
    for (size_t block = 0; block < blockmodel.num_blocks(); ++block) {
        long block_weight = get_weight(neighbor_block, block, blockmodel);
        cumulative_weight += block_weight;
        bool first = (chosen < 0) & (cumulative_weight > r);
        chosen = first ? block : chosen;  // if chosen = -1, and cumulative_weight > r, then choose this neighbor
    }
    return chosen; // something went VERY wrong here
    
}

// TODO: get rid of block_assignment, just use blockmodel?
ProposedMove propose_new_block(long vertex, long current_block, const CSR &graph_csr,
                               const CSR &graph_csc, const BlockmodelGPUView &blockmodel, pcg32 &rng) {
    long k_out = 0, k_in = 0;
    NeighborView out_neighbors = graph_csr.neighbors(vertex);
    NeighborView in_neighbors = graph_csc.neighbors(vertex);
    for (size_t i = 0; i < out_neighbors.size(); ++i) {
        k_out += out_neighbors.val(i);
    }
    for (size_t i = 0; i < in_neighbors.size(); ++i) {
        k_in += in_neighbors.val(i) * (long)(in_neighbors[i] != vertex); // do not count self-edges
    }
    // std::vector<long> neighbor_indices = utils::concatenate<long>(out_blocks.indices, in_blocks.indices);
    // std::vector<long> neighbor_weights = utils::concatenate<long>(out_blocks.values, in_blocks.values);
    long k = k_out + k_in;
    long num_blocks = blockmodel.num_blocks();

    // If the current vertex has no neighbors, propose merge with random block
    if (k == 0) {
       long proposal = propose_random_block(current_block, num_blocks, rng);
       return ProposedMove{vertex, current_block, proposal, k_out, k_in, k};
    //    return utils::ProposalAndEdgeCounts{proposal, k_out, k_in, k};
    }
    // Choose a neighbor vertex
    long neighbor = choose_neighbor(vertex, k, out_neighbors, in_neighbors, rng);
    // Get the block of the neighbor vertex
    long neighbor_block = blockmodel.block_assignment(neighbor);

    long dense_total = blockmodel.degrees(neighbor_block);
    // long dense_total = 0;
    // #pragma omp parallel for reduction(+:dense_total)
    // for (size_t i = 0; i < num_blocks; ++i) {
        // dense_total += get_weight(neighbor_block, i, blockmodel);
    // }
    if (dense_total == 0) { // Neighbor's block has no usable neighbors, so propose a random block
        long proposal = propose_random_block(current_block, num_blocks, rng);
        return ProposedMove{vertex, current_block, proposal, k_out, k_in, k};
        // return utils::ProposalAndEdgeCounts{proposal, k_out, k_in, k};
    }
    // Choose a second-degree neighbor block as the proposal
    long proposal = choose_neighbor(neighbor_block, blockmodel, dense_total, rng);
    return ProposedMove{vertex, current_block, proposal, k_out, k_in, k};
    // return utils::ProposalAndEdgeCounts{proposal, k_out, k_in, k};
}

// TODO: get rid of block_assignment, just use blockmodel?
// utils::ProposalAndEdgeCounts propose_new_block_block_merge(long current_block, const CSR &graph_csr, const CSR &graph_csc,
//                                                            const BlockmodelGPUView &blockmodel) {
//     std::vector<long> neighbor_indices = utils::concatenate<long>(out_blocks.indices, in_blocks.indices);
//     std::vector<long> neighbor_weights = utils::concatenate<long>(out_blocks.values, in_blocks.values);
//     long k_out = std::accumulate(out_blocks.values.begin(), out_blocks.values.end(), 0);
//     long k_in = std::accumulate(in_blocks.values.begin(), in_blocks.values.end(), 0);
//     long k = k_out + k_in;
//     long num_blocks = blockmodel.num_blocks();

//     // If the current block has no neighbors, propose merge with random block
//     if (k == 0) {
//         long proposal = propose_random_block(current_block, num_blocks);
//         return utils::ProposalAndEdgeCounts{proposal, k_out, k_in, k};
//     }
//     long neighbor_block = choose_neighbor(neighbor_indices, neighbor_weights);

//     // Dense compute path: build the neighbor-weight multinomial directly from
//     // dense getrow/getcol vectors (indexed by block id) instead of materializing
//     // a MapVector via neighbors_weights.
//     const std::shared_ptr<ISparseMatrix> matrix = blockmodel.blockmatrix();
//     std::vector<long> row = matrix->getrow(neighbor_block);
//     std::vector<long> col = matrix->getcol(neighbor_block);
//     std::vector<long> dense_edges(num_blocks, 0);
//     // Mirror DenseMatrix::neighbors_weights: outgoing edges from the row
//     // (self-edge included), incoming edges from the column excluding the
//     // diagonal to avoid double counting.
//     for (long i = 0; i < num_blocks; ++i) {
//       dense_edges[i] = row[i] + (i != neighbor_block ? col[i] : 0);
//     }
//     dense_edges[current_block] = 0;
//     long dense_total = utils::sum<long>(dense_edges);
//     if (dense_total == 0) { // Neighbor block has no usable neighbors, so propose a random block
//         long proposal = propose_random_block(current_block, num_blocks);
//         return utils::ProposalAndEdgeCounts{proposal, k_out, k_in, k};
//     }
//     long proposal = choose_neighbor(dense_edges);
//     return utils::ProposalAndEdgeCounts{proposal, k_out, k_in, k};
// }

long propose_random_block(long current_block, long num_blocks, pcg32 &rng) {
    long proposed = random_integer(num_blocks - 1, rng);
    proposed += 1 * (proposed >= current_block);
    return proposed;
}

#pragma omp end declare target

}

long propose_random_block(long current_block, long num_blocks) {
    // Generate numbers 0.._num_blocks-2 in order to exclude the current block
    // std::uniform_int_distribution<long> distribution(0, _num_blocks - 2);
    // long proposed = distribution(rng::generator());
    long proposed = candidates(rng::generator());
    if (proposed >= current_block) {
        proposed++;
    }
    return proposed;
}

long random_integer(long low, long high) {
    std::uniform_int_distribution<long> distribution(low, high);
    return distribution(rng::generator());
}

namespace directed {

double delta_entropy_temp(std::vector<long> &row_or_col, std::vector<long> &block_degrees, long degree) {
    // std::cout << "dE_temp_directed_dense!" << std::endl;
    std::vector<double> row_or_col_double = utils::to_double<long>(row_or_col);
    std::vector<double> block_degrees_double = utils::to_double<long>(block_degrees);
    std::vector<double> result = row_or_col_double / block_degrees_double / double(degree);
    result = row_or_col_double * utils::nat_log<double>(result);
    return (double)utils::sum<double>(result);
}

double delta_entropy_temp(const MapVector<long> &row_or_col, const std::vector<long> &block_degrees, long degree) {
    // std::cout << "dE_temp_directed_sparse!" << std::endl;
    // throw std::runtime_error("SHOULD BE UNDIRECTED");
    double result = 0.0;
    for (const auto &pair : row_or_col) {
        if (pair.second == 0)  // 0s sometimes get inserted into the sparse matrix
            continue;
        double temp = (double) pair.second / (double) block_degrees[pair.first] / degree;
        temp = (double) pair.second * std::log(temp);
        result += temp;
    }
    return result;
}

double delta_entropy_temp(const MapVector<long> &row_or_col, const std::vector<long> &block_degrees, long degree,
                          long current_block, long proposal) {
    // std::cout << "dE_temp_directed_sparse_ignore!" << std::endl;
    // throw std::runtime_error("SHOULD BE UNDIRECTED");
    double result = 0.0;
    for (const auto &pair : row_or_col) {
        // 0s sometimes get inserted into the sparse matrix
        if (pair.second == 0 || pair.first == current_block || pair.first == proposal)
            continue;
        double temp = (double) pair.second / (double) block_degrees[pair.first] / degree;
        temp = (double) pair.second * std::log(temp);
        result += temp;
    }
    return result;
}
}  // namespace directed

namespace undirected {

double delta_entropy_temp(std::vector<long> &row_or_col, std::vector<long> &block_degrees, long degree, long num_edges) {
    // std::cout << "dE_temp_undirected_dense!" << std::endl;
    std::vector<double> row_or_col_double = utils::to_double<long>(row_or_col) / 2.0;
    std::vector<double> block_degrees_double = utils::to_double<long>(block_degrees) / 2.0;
    std::vector<double> result = (row_or_col_double / 2.0) / (block_degrees_double * degree);
    // std::vector<double> result = row_or_col_double / (block_degrees_double * degree * 2.0 * num_edges);
    result = row_or_col_double * utils::nat_log<double>(result);
    double dE = 0.5 * utils::sum<double>(result);
    assert(!std::isnan(dE));
    return dE;
    // std::vector<double> row_or_col_double = utils::to_double<long>(row_or_col) / (2.0 * num_edges);
    // std::vector<double> block_degrees_double = utils::to_double<long>(_block_degrees) / (2.0 * num_edges);
    // std::vector<double> result = (row_or_col_double * 2.0 * num_edges) / (block_degrees_double * degree);
    // // std::vector<double> result = row_or_col_double / (block_degrees_double * degree * 2.0 * num_edges);
    // result = row_or_col_double * utils::nat_log<double>(result);
    // double dE = 0.5 * utils::sum<double>(result);
    // assert(!std::isnan(dE));
    // return dE;
    // return 0.5 * utils::sum<double>(result);
}

double delta_entropy_temp(const MapVector<long> &row_or_col, const std::vector<long> &block_degrees, long degree,
                          long num_edges) {
    double result = 0.0;
    double deg = degree / 2.0;
    for (const std::pair<long, long> &pair : row_or_col) {
        if (pair.second == 0)  // 0s sometimes get inserted into the sparse matrix
            continue;
        double block_deg = (double) block_degrees[pair.first] / 2.0;
        // double temp = (double) pair.second / (2.0 * num_edges * (double) _block_degrees[pair.first] * degree);
        double temp = ((double) pair.second / 2.0) / (block_deg * deg);
        temp = (double) pair.second * std::log(temp);
        result += temp;
    }
    result *= 0.5;
    assert(!std::isnan(result));
    return result;
    // // std::cout << "dE_temp_undirected_sparse!" << std::endl;
    // double result = 0.0;
    // double deg = degree / (2.0 * num_edges);
    // for (const std::pair<long, long> &pair : row_or_col) {
    //     if (pair.second == 0)  // 0s sometimes get inserted into the sparse matrix
    //         continue;
    //     double block_deg = (double) _block_degrees[pair.first] / (2.0 * num_edges);
    //     // double temp = (double) pair.second / (2.0 * num_edges * (double) _block_degrees[pair.first] * degree);
    //     double temp = ((double) pair.second * 2.0 * num_edges) / (block_deg * deg);
    //     temp = (double) pair.second * std::log(temp);
    //     result += temp;
    // }
    // result *= 0.5;
    // assert(!std::isnan(result));
    // return result;
}

double delta_entropy_temp(const MapVector<long> &row_or_col, const std::vector<long> &block_degrees, long degree,
                          long current_block, long proposal, long num_edges) {
    double result = 0.0;
    double deg = degree / 2.0;
    for (const std::pair<long, long> &pair : row_or_col) {
        // 0s sometimes get inserted into the sparse matrix
        if (pair.second == 0 || pair.first == current_block || pair.first == proposal)
            continue;
        double block_deg = (double) block_degrees[pair.first] / 2.0;
        double temp = ((double) pair.second / 2.0) / (block_deg * deg);
        // double temp = (double) pair.second / (2.0 * num_edges * (double) _block_degrees[pair.first] * degree);
        temp = (double) pair.second * std::log(temp);
        result += temp;
    }
    result *= 0.5;
    assert(!std::isnan(result));
    return result;
    // double result = 0.0;
    // double deg = degree / (2.0 * num_edges);
    // for (const std::pair<long, long> &pair : row_or_col) {
    //     // 0s sometimes get inserted into the sparse matrix
    //     if (pair.second == 0 || pair.first == current_block || pair.first == proposal)
    //         continue;
    //     double block_deg = (double) _block_degrees[pair.first] / (2.0 * num_edges);
    //     double temp = ((double) pair.second * 2.0 * num_edges) / (block_deg * deg);
    //     // double temp = (double) pair.second / (2.0 * num_edges * (double) _block_degrees[pair.first] * degree);
    //     temp = (double) pair.second * std::log(temp);
    //     result += temp;
    // }
    // result *= 0.5;
    // assert(!std::isnan(result));
    // return result;
}
}  // namespace undirected

}  // namespace common
