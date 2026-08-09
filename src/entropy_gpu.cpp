#include "entropy.hpp"

#include "blockmodel.hpp"
#include "matrix/csr.hpp"
#include "blockmodel/delta_coo.hpp"
#include "typedefs.hpp"
#include "utils.hpp"

namespace entropy {

namespace gpu {

// double hastings_correction(long vertex, const CSR &graph_csr, const CSR &graph_csc, const BlockmodelGPUView &blockmodel,
//                             const DeltaCOO &delta, long current_block, const utils::ProposalAndEdgeCounts &proposal) {
//     if (proposal.num_neighbor_edges == 0 || !args.hastings_correction || !args.parametric) {  // No correction if disabled or in nonparametric mode
//         return 1.0;
//     }  // on the gpu, we're always doing hastings correction
//     // Compute block weights
//     // MapVector<long> block_counts;
//     long* block_counts = new long[blockmodel.num_blocks()]();
//     for (const long neighbor: graph_csr.neighbors(vertex)) {
//         long neighbor_block = blockmodel.block_assignment(neighbor);
//         block_counts[neighbor_block] += 1;
//     }
//     for (const long neighbor: graph_csc.neighbors(vertex)) {
//         long neighbor_block = blockmodel.block_assignment(neighbor);
//         block_counts[neighbor_block] += 1 * (neighbor != vertex);
//     }
//     // Create Arrays using unique blocks
//     size_t num_unique_blocks = 0;
//     for (size_t i = 0; i < blockmodel.num_blocks(); ++i) {
//         num_unique_blocks += 1 * (block_counts[i] > 0);
//     }
//     double* counts = new double[num_unique_blocks]();
//     double* proposal_weights = new double[num_unique_blocks]();
//     double* block_weights = new double[num_unique_blocks]();
//     double* block_degrees = new double[num_unique_blocks]();
//     double* proposal_degrees = new double[num_unique_blocks]();
//     // Fill Arrays
//     long index = 0;
//     long num_blocks = blockmodel.num_blocks();
//     // TODO: the arrays above can be removed, since p_forward and p_backward can be computed on the fly.
//     // this is because the computations only occur at current index.
//     for (size_t i = 0; i < blockmodel.num_blocks(); ++i) {
//         if (block_counts[i] == 0) continue;
//         counts[index] = block_counts[i];
//         double proposal_row_val = (double) blockmodel.get(proposal.proposal, i);
//         double proposal_col_val = (double) blockmodel.get(i, proposal.proposal); 
//         proposal_weights[index] = proposal_row_val + proposal_col_val + 1.0;
//         block_degrees[index] = blockmodel.degrees(i) + num_blocks;
//         block_weights[index] = blockmodel.get(current_block, i) +
//                                 delta.get(current_block, i) +
//                                 //                get(delta, std::make_pair(current_block, entry.first)) +
//                                 blockmodel.get(i, current_block) +
//                                 delta.get(i, current_block) + 1.0;
// //                get(delta, std::make_pair(entry.first, current_block)) + 1.0;
//         long new_block_degree = blockmodel.degrees(i);
//         if (i == current_block) {
//             long current_block_self_edges = blockmodel.get(current_block, current_block)
//                                             + delta.get(current_block, current_block);
//             long degree_out = blockmodel.degrees_out(current_block) - proposal.num_out_neighbor_edges;
//             long degree_in = blockmodel.degrees_in(current_block) - proposal.num_in_neighbor_edges;
//             new_block_degree = degree_out + degree_in - current_block_self_edges;
//         } else if (i == proposal.proposal) {
//             long proposed_block_self_edges = blockmodel.get(proposal.proposal, proposal.proposal)
//                                             + delta.get(proposal.proposal, proposal.proposal);
//             long degree_out = blockmodel.degrees_out(proposal.proposal) + proposal.num_out_neighbor_edges;
//             long degree_in = blockmodel.degrees_in(proposal.proposal) + proposal.num_in_neighbor_edges;
//             new_block_degree = degree_out + degree_in - proposed_block_self_edges;
//         }
// //        proposal_degrees[index] = new_block_degrees.block_degrees[entry.first] + _num_blocks;
//         proposal_degrees[index] = new_block_degree + num_blocks;
//         index++;
//     }
//     // Compute p_forward and p_backward
//     // auto p_forward = utils::sum<double>(counts * proposal_weights / block_degrees);
//     // auto p_backward = utils::sum<double>(counts * block_weights / proposal_degrees);
//     double p_forward = 0;
//     double p_backward = 0;
//     for (size_t i = 0; i < num_unique_blocks; ++i) {
//         p_forward += counts[i] * proposal_weights[i] / block_degrees[i];
//         p_backward += counts[i] * block_weights[i] / proposal_degrees[i];
//     }
//     delete[] block_counts;
//     delete[] counts;
//     delete[] proposal_weights;
//     delete[] block_weights;
//     delete[] block_degrees;
//     delete[] proposal_degrees;
//     return p_backward / p_forward;
// }

namespace nonparametric {

#pragma omp declare target

double delta_mdl(const BlockmodelGPUView &blockmodel, const CSR &graph_csr, const CSR &graph_csc,
                 const ProposedMove &proposal) {
//    std::cout << blockmodel.block_assignment(vertex) << " != " << delta.current_block() << std::endl;
    assert(blockmodel.block_assignment(proposal.vertex) == proposal.current_block);

//    get_move_entries(v, r, nr, m_entries, [](auto) constexpr { return false; });

    if (proposal.current_block == proposal.proposed_block) return 0;
//    if (r == nr || _vweight[v] == 0)
//        return 0;

    double dS = 0;
    dS = virtual_move_sparse(blockmodel, graph_csr, graph_csc, proposal, 1);  // <true>(v, r, nr, m_entries);

    double dS_dl = 0;

    dS_dl += get_delta_partition_dl(graph_csr.num_rows(), blockmodel, proposal, 1);  // v, r, nr, ea);
    assert(!std::isinf(dS_dl));
    assert(!std::isnan(dS_dl));

//    if (ea.degree_dl || ea.edges_dl) {
//    auto& ps = get_partition_stats(v);
//    if (_deg_corr && ea.degree_dl)
    dS_dl += 0.0; // currently only handling nonparametric case, where degree corrected = false.
    // ds_dl += get_delta_deg_dl(vertex, blockmodel, delta, graph_csr, graph_csc);  // v, r, nr, _vweight, _eweight, _degs, _g, ea.degree_dl_kind);
//    if (ea.edges_dl)
//    {
//    size_t actual_B = 0;
//    for (auto& ps : _partition_stats)
//        actual_B += ps.get_actual_B();
    dS_dl += get_delta_edges_dl(blockmodel, proposal, 1, graph_csr.nnz());  // v, r, nr, _vweight, actual_B, _g);

    return dS + BETA_DL * dS_dl;
}

//template <class Graph, class VProp, class EProp, class Degs>
// double get_delta_deg_dl(long vertex, const BlockmodelGPUView &blockmodel, const DeltaCOO &delta, const CSR &graph_csr, const CSR &graph_csc) {  // size_t r, size_t nr, VProp& vweight,
//     if (!args.degreecorrected) return 0.00;
//     if (delta.current_block() == delta.proposed_block()) return 0.;  // for block_merge, it's this || size(block) == 0

//     long vkin = graph.in_neighbors(vertex).size();
//     long vkout = graph.out_neighbors(vertex).size();
//     double dS = 0;

//     dS += get_delta_deg_dl_dist_change(blockmodel, delta.current_block(),  vkin, vkout, 1, -1);

//     dS += get_delta_deg_dl_dist_change(blockmodel, delta.proposed_block(), vkin, vkout, 1, +1);
//     assert(!std::isinf(dS));
//     assert(!std::isnan(dS));

//     return dS;
// }

//template <class VProp, class Graph>
double get_delta_edges_dl(const BlockmodelGPUView &blockmodel, const ProposedMove &proposal, long weight, long num_edges) {
    if (proposal.current_block == proposal.proposed_block)
        return 0;

    double S_b = 0, S_a = 0;

    int dB = 0;
    dB -= 1 * (blockmodel.block_size(proposal.current_block) == weight);
    dB += 1 * (blockmodel.block_size(proposal.proposed_block) == 0);

    if (dB != 0) {
        S_b += get_edges_dl(blockmodel.num_nonempty_blocks(), num_edges);
        S_a += get_edges_dl(blockmodel.num_nonempty_blocks() + dB, num_edges);
    }

    double dS = S_a - S_b;
    assert(!std::isinf(dS));
    assert(!std::isnan(dS));
    return dS;
}

double get_delta_partition_dl(long num_vertices, const BlockmodelGPUView &blockmodel, const ProposedMove &proposal, long weight) {  // size_t v, size_t r, size_t nr, const entropy_args_t& ea) {
    if (proposal.current_block == proposal.proposed_block) return 0.;

    double dS = 0;
    double S_b = 0;
    double S_a = 0;

    // Use gpu_lgamma (OCML on device) instead of the cached host fastlgamma so this stays device-safe on the GPU offload path.
    S_b += -gpu_lgamma(blockmodel.block_size(proposal.current_block) + 1);  // _total[r] + 1);
    S_a += -gpu_lgamma(blockmodel.block_size(proposal.current_block) - weight + 1);  // _total[r] - n + 1);

    S_b += -gpu_lgamma(blockmodel.block_size(proposal.proposed_block) + 1);  // _total[nr] + 1);
    S_a += -gpu_lgamma(blockmodel.block_size(proposal.proposed_block) + weight + 1);  // _total[nr] + n + 1);

    int dB = 0;
    dB -= 1 * (blockmodel.block_size(proposal.current_block) == weight);
    dB += 1 * (blockmodel.block_size(proposal.proposed_block) == 0);

    if (dB != 0) {
        S_b += fastlbinom(num_vertices - 1, blockmodel.num_nonempty_blocks() - 1);
        S_a += fastlbinom(num_vertices - 1, blockmodel.num_nonempty_blocks() + dB - 1);
    }

    dS += S_a - S_b;
    assert(!std::isinf(dS));
    assert(!std::isnan(dS));

    return dS;
}

double get_edges_dl(size_t B, size_t E) {
    // size_t NB = !args.undirected ? B * B : (B * (B + 1)) / 2;
    size_t NB = B * B; // only handling undirected case for now
    double E_dl = fastlbinom(NB + E - 1, E);
//    std::cout << "edges_dl: " << E_dl << std::endl;
    return E_dl;
}

/// Sum of non-self edges incident to `block`
long incident_edges(long vertex, const NeighborView &neighbors, const BlockmodelGPUView &blockmodel, long block) {
    long edges = 0;
    for (long index = 0; index < neighbors.size(); ++index) {
        long neighbor = neighbors[index];
        long not_self_edge = (neighbor != vertex);
        long correct_block = (blockmodel.block_assignment(neighbor) == block);
        edges += neighbors.val(index) * not_self_edge * correct_block;
    }
    return edges;
}

/// Sum of self-edge weights
long self_edge_weight(const NeighborView &out_neighbors, long vertex) {
    long edges = 0;
    for (long index = 0; index < out_neighbors.size(); ++index) {
        long neighbor = out_neighbors[index];
        edges += out_neighbors.val(index) * (neighbor == vertex);
    }
    return edges;
}

long cell_change(const ProposedMove &proposal, long self_edges, const BlockmodelGPUView &blockmodel, long row, long col,
                 long out_edges, long in_edges) {
    long change = 0;
    change += out_edges * (row == proposal.proposed_block);
    change += in_edges * (col == proposal.proposed_block);
    change -= out_edges * (row == proposal.current_block);
    change -= in_edges * (col == proposal.current_block);
    change += self_edges * (row == proposal.proposed_block && col == proposal.proposed_block);
    change -= self_edges * (row == proposal.current_block && col == proposal.current_block);
    return change;
}

// long cell_change(const ProposedMove &proposal, long self_edges, const BlockmodelGPUView &blockmodel, long row, long col,
//                  const NeighborView &out_neighbors, const NeighborView &in_neighbors) {
//     long change = 0;
//     long out_edges = incident_edges(proposal.vertex, out_neighbors, blockmodel, col);
//     long in_edges = incident_edges(proposal.vertex, in_neighbors, blockmodel, row);
//     change += out_edges * (row == proposal.proposed_block);
//     change += in_edges * (col == proposal.proposed_block);
//     change -= out_edges * (row == proposal.current_block);
//     change -= in_edges * (col == proposal.current_block);
//     change += self_edges * (row == proposal.proposed_block && col == proposal.proposed_block);
//     change -= self_edges * (row == proposal.current_block && col == proposal.current_block);
//     return change;
// }

// obtain the entropy difference given a set of entries in the e_rs matrix
//template <bool exact, class MEntries, class Eprop, class EMat, class BGraph>
//[[gnu::always_inline]] [[gnu::flatten]] [[gnu::hot]] inline
double entries_dS(const BlockmodelGPUView &blockmodel, const CSR &graph_csr, const CSR &graph_csc,
                  const ProposedMove &proposal) {  // MEntries& m_entries, Eprop& mrs, EMat& emat, BGraph& bg) {
    double dS = 0;
    using gpu::nonparametric::eterm_exact;
    NeighborView out_neighbors = graph_csr.neighbors(proposal.vertex);
    NeighborView in_neighbors = graph_csc.neighbors(proposal.vertex);
    long self_edges = self_edge_weight(out_neighbors, proposal.vertex);

    long start_ptr = graph_csr.row_ptrs[proposal.vertex];
    long end_ptr = start_ptr + graph_csr.degree(proposal.vertex);
    long out_weight_current = 0, out_weight_proposed = 0, in_weight_current = 0, in_weight_proposed = 0;
    while (start_ptr < end_ptr) {
        long block = graph_csr._block_id[start_ptr];
        long out_weight = 0;
        while (start_ptr < end_ptr && graph_csr._block_id[start_ptr] == block) {
            out_weight += graph_csr._edge_weight[start_ptr];
            ++start_ptr;
        }
        out_weight_current += (block == proposal.current_block) * (out_weight - self_edges);
        out_weight_proposed += (block == proposal.proposed_block) * out_weight;
        if (block == proposal.current_block || block == proposal.proposed_block) continue;
        // dS for (current_block, block)
        long change = cell_change(proposal, self_edges, blockmodel, proposal.current_block, block, out_weight, 0);
        auto value = (long) blockmodel.get(proposal.current_block, block);
        dS += eterm_exact(proposal.current_block, block, value + change) - eterm_exact(proposal.current_block, block, value);
        // dS for (proposed_block, block)
        change = cell_change(proposal, self_edges, blockmodel, proposal.proposed_block, block, out_weight, 0);
        value = (long) blockmodel.get(proposal.proposed_block, block);
        dS += eterm_exact(proposal.proposed_block, block, value + change) - eterm_exact(proposal.proposed_block, block, value);
        assert(!std::isinf(dS));
        assert(!std::isnan(dS));
    }

    start_ptr = graph_csc.row_ptrs[proposal.vertex];
    end_ptr = start_ptr + graph_csc.degree(proposal.vertex);

    while (start_ptr < end_ptr) {
        long block = graph_csc._block_id[start_ptr];
        long in_weight = 0;
        while (start_ptr < end_ptr && graph_csc._block_id[start_ptr] == block) {
            in_weight += graph_csc._edge_weight[start_ptr];
            ++start_ptr;
        }
        in_weight_current += (block == proposal.current_block) * (in_weight - self_edges);
        in_weight_proposed += (block == proposal.proposed_block) * in_weight;
        if (block == proposal.current_block || block == proposal.proposed_block) continue;
        // dS for (block, current_block)
        long change = cell_change(proposal, self_edges, blockmodel, block, proposal.current_block, 0, in_weight);
        auto value = (long) blockmodel.get(block, proposal.current_block);
        dS += eterm_exact(block, proposal.current_block, value + change) - eterm_exact(block, proposal.current_block, value);
        // dS for (block, proposed_block)
        change = cell_change(proposal, self_edges, blockmodel, block, proposal.proposed_block, 0, in_weight);
        value = (long) blockmodel.get(block, proposal.proposed_block);
        dS += eterm_exact(block, proposal.proposed_block, value + change) - eterm_exact(block, proposal.proposed_block, value);
        assert(!std::isinf(dS));
        assert(!std::isnan(dS));
    }
    // handle (current_block, current_block)
    long change = cell_change(proposal, self_edges, blockmodel, proposal.current_block, proposal.current_block, out_weight_current, in_weight_current);
    auto value = (long) blockmodel.get(proposal.current_block, proposal.current_block);
    dS += eterm_exact(proposal.current_block, proposal.current_block, value + change) - eterm_exact(proposal.current_block, proposal.current_block, value);
    // handle (current_block, proposed_block)
    change = cell_change(proposal, self_edges, blockmodel, proposal.current_block, proposal.proposed_block, out_weight_proposed, in_weight_current);
    value = (long) blockmodel.get(proposal.current_block, proposal.proposed_block);
    dS += eterm_exact(proposal.current_block, proposal.proposed_block, value + change) - eterm_exact(proposal.current_block, proposal.proposed_block, value);
    // handle (proposed_block, current_block)
    change = cell_change(proposal, self_edges, blockmodel, proposal.proposed_block, proposal.current_block, out_weight_current, in_weight_proposed);
    value = (long) blockmodel.get(proposal.proposed_block, proposal.current_block);
    dS += eterm_exact(proposal.proposed_block, proposal.current_block, value + change) - eterm_exact(proposal.proposed_block, proposal.current_block, value);
    // handle (proposed_block, proposed_block)
    change = cell_change(proposal, self_edges, blockmodel, proposal.proposed_block, proposal.proposed_block, out_weight_proposed, in_weight_proposed);
    value = (long) blockmodel.get(proposal.proposed_block, proposal.proposed_block);
    dS += eterm_exact(proposal.proposed_block, proposal.proposed_block, value + change) - eterm_exact(proposal.proposed_block, proposal.proposed_block, value);
    // // walk through affected blockmodel rows and columns, compute the change in entropy for each cell
    // #pragma omp parallel for reduction(+:dS)
    // for (long index = 0; index < blockmodel.num_blocks(); ++index) {
    //     long col = index;
    //     for (long row : {proposal.current_block, proposal.proposed_block}) {
    //         long change = cell_change(proposal, self_edges, blockmodel, row, col, out_neighbors, in_neighbors);
    //         auto value = (long) blockmodel.get(row, col);
    //         dS += eterm_exact(row, col, value + change) - eterm_exact(row, col, value);
    //         assert(!std::isinf(dS));
    //         assert(!std::isnan(dS));
    //     }
    //     long row = index;
    //     if (row == proposal.current_block || row == proposal.proposed_block) continue;
    //     for (long col : {proposal.current_block, proposal.proposed_block}) {
    //         long change = cell_change(proposal, self_edges, blockmodel, row, col, out_neighbors, in_neighbors);
    //         auto value = (long) blockmodel.get(row, col);
    //         dS += eterm_exact(row, col, value + change) - eterm_exact(row, col, value);
    //         assert(!std::isinf(dS));
    //         assert(!std::isnan(dS));
    //     }
    // }
    
    return dS;
}

// compute the entropy difference of a virtual move of vertex from block r
// to nr
//template <bool exact, class MEntries>
double virtual_move_sparse(const BlockmodelGPUView &blockmodel, const CSR &graph_csr, const CSR &graph_csc,
                           const ProposedMove &proposal, long weight) {  // size_t v, size_t r, size_t nr, MEntries& m_entries) {
    // TODO: see if I can safely get rid of this branch
    if (proposal.current_block == proposal.proposed_block) return 0.;

    double dS = entries_dS(blockmodel, graph_csr, graph_csc, proposal);  // <exact>(m_entries, _mrs, _emat, _bg);

    long kin = proposal.num_in_neighbor_edges;
    long kout = proposal.num_out_neighbor_edges;

    using gpu::nonparametric::vterm_exact;
    // auto vt = [&](auto out_degree, auto in_degree, auto w) { // , auto nr) {
    //     return gpu::nonparametric::vterm_exact(out_degree, in_degree, w);  // , nr, _deg_corr, _bg);
    // };

    dS += vterm_exact(blockmodel.degrees_out(proposal.current_block) - kout, blockmodel.degrees_in(proposal.current_block) - kin, blockmodel.block_size(proposal.current_block) - weight);  // , wr_r - dwr);
    dS -= vterm_exact(blockmodel.degrees_out(proposal.current_block), blockmodel.degrees_in(proposal.current_block), blockmodel.block_size(proposal.current_block));  //        , mrm_r      , wr_r      );
    assert(!std::isinf(dS));
    assert(!std::isnan(dS));

    dS += vterm_exact(blockmodel.degrees_out(proposal.proposed_block) + kout, blockmodel.degrees_in(proposal.proposed_block) + kin, blockmodel.block_size(proposal.proposed_block) + weight);  // , wr_nr + dwnr);
    dS -= vterm_exact(blockmodel.degrees_out(proposal.proposed_block), blockmodel.degrees_in(proposal.proposed_block), blockmodel.block_size(proposal.proposed_block));  //        , mrm_nr      , wr_nr       );
    assert(!std::isinf(dS));
    assert(!std::isnan(dS));

    return dS;
}

#pragma omp end declare target

}  // namespace nonparametric

}  // namespace gpu

} // namespace entropy
