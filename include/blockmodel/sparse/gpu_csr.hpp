/***
 * Sparse Matrix in CSR format, written for GPU offloading.
 */
#ifndef SBP_BLOCKMODEL_SPARSE_CSR_MATRIX_HPP
#define SBP_BLOCKMODEL_SPARSE_CSR_MATRIX_HPP

#include <vector>
#include "typedefs.hpp"

/**
 * The CSR format sparse matrix. Made up of 3 lists: a list of values, 
 * a list of row pointers, and a list of column indices.
 */
class CSRMatrix {
  public:
    CSRMatrix() = default;
    CSRMatrix(long nrows, long ncols, long buckets = 10) {
        throw std::runtime_error("rows, cols constructor impossible to implement for CSR matrix!");
    }
    /// Constructor for making a matrix to store an adjacency list
    CSRMatrix(const NeighborList &neighbor_list, uint num_vertices, uint num_edges) {
        this->vals = (long *)std::malloc(num_edges * sizeof(long));
        this->row_ptrs = (uint *)std::malloc((num_vertices + 1) * sizeof(uint));
        this->col_indices = (uint *)std::malloc(num_edges * sizeof(uint));
        uint index = 0;
        for (uint row = 0; row < num_vertices; ++row) {
            const std::vector<long> &row_vector = neighbor_list[row];
            this->row_ptrs[row] = index;
            for (const long &col : row_vector) {
                this->vals[index] = 1;
                this->col_indices[index] = (uint) col;
                index++;
            }
            this->row_ptrs[num_vertices] = index; 
        }
        this->nrows = num_vertices;
        this->nnz = num_edges;
    }
    ~CSRMatrix() {
        std::free(this->vals);
        std::free(this->row_ptrs);
        std::free(this->col_indices);
    }
    uint degree(uint index) const {
        return this->row_ptrs[index + 1] - this->row_ptrs[index];
    }
    // void add(long row, long col, long val) override;
    // void add_transpose(long row, long col, long val);
    // /// Clears the value in a given row. Complexity ~O(number of blocks).
    // void clearrow(long row) override;
    // /// Clears the values in a given column. Complexity ~O(number of blocks).
    // void clearcol(long col) override;
    // /// Returns a copy of the current matrix.
    // ISparseMatrix* copy() const override;
    // long distinct_edges(long block) const override;
    // std::vector<std::tuple<long, long, long>> entries() const override;
    // long get(long row, long col) const override;
    // /// Returns all values in the requested column as a dense vector.
    // std::vector<long> getcol(long col) const override;
    // /// Returns all values in the requested column as a sparse vector (ordered map).
    // MapVector<long> getcol_sparse(long col) const override;
    // const MapVector<long>& getcol_sparseref(long col) const override;
    // void getcol_sparse(long col, MapVector<long> &col_vector) const override;
    // /// Returns all values in the requested row as a dense vector.
    // std::vector<long> getrow(long row) const override;
    // /// Returns all values in the requested column as a sparse vector (ordered map).
    // MapVector<long> getrow_sparse(long row) const override;
    // const MapVector<long>& getrow_sparseref(long row) const override;
    // void getrow_sparse(long row, MapVector<long> &row_vector) const override;
    // EdgeWeights incoming_edges(long block) const override;
    // std::set<long> neighbors(long block) const override;
    // MapVector<long> neighbors_weights(long block) const override;
    // Indices nonzero() const override;
    // long nonzeros() const override;
    // EdgeWeights outgoing_edges(long block) const override;
    // /// Sets the values in a row equal to the input vector.
    // void setrow(long row, const MapVector<long> &vector) override;
    // /// Sets the values in a column equal to the input vector.
    // void setcol(long col, const MapVector<long> &vector) override;
    // void sub(long row, long col, long val) override;
    // long edges() const override;
    // void print() const override;
    // std::vector<long> sum(long axis = 0) const override;
    // long trace() const override;
    // void update_edge_counts(long current_block, long proposed_block, std::vector<long> current_row,
    //                                 std::vector<long> proposed_row, std::vector<long> current_col,
    //                                 std::vector<long> proposed_col) override;
    // void update_edge_counts(long current_block, long proposed_block, MapVector<long> current_row,
    //                         MapVector<long> proposed_row, MapVector<long> current_col,
    //                         MapVector<long> proposed_col) override;
    // void update_edge_counts(const Delta &delta) override;
    // bool validate(long row, long col, long val) const override;
    // std::vector<long> values() const override;

//   private:
    // void check_row_bounds(long row);
    // void check_col_bounds(long col);
    // void check_row_bounds(long row) const;
    // void check_col_bounds(long col) const;
    // long ncols;
    // long nrows;
//    std::vector<std::unordered_map<long, long>> matrix;
//    std::vector<std::unordered_map<long, long>> matrix_transpose;
    long* vals;
    uint* row_ptrs;
    uint* col_indices;
    long nrows;
    long nnz;
};

#endif // SBP_BLOCKMODEL_SPARSE_CSR_MATRIX_HPP
