/**
 * Functions for porting the CPU SBP code to the GPU. Currently only contains functions related to the MCMC phase.
 */
#ifndef SBP_BLOCKMODEL_SPARSE_CSR_MATRIX_HPP
#define SBP_BLOCKMODEL_SPARSE_CSR_MATRIX_HPP

#include <vector>
#include "blockmodel.hpp"
#include "gpu_csr.hpp"
#include "typedefs.hpp"

/**
 * The CSR format Blockmodel.
 */
class GPUBlockmodel {
  public:
    GPUBlockmodel() = default;
    GPUBlockmodel(const Blockmodel &blockmodel) {

    }
    
};

#endif // SBP_BLOCKMODEL_SPARSE_CSR_MATRIX_HPP