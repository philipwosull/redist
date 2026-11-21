// sparse_logdet.h
#pragma once

#include <memory>
#include <vector>

// Minimal stand-in for Eigen::Triplet: one instance per nonzero, same layout.
//
// Eigen is deliberately kept out of this header. <Eigen/Sparse> is expensive to
// parse and this header is reached by map_calc.h, which nearly every
// translation unit includes. Callers build entries with this type and Eigen
// stays confined to sparse_logdet.cpp.
struct SparseEntry {
    int row;
    int col;
    double value;

    SparseEntry(int const row, int const col, double const value)
        : row(row), col(col), value(value) {}
};

// Log determinant of a sparse symmetric positive definite matrix, given only
// its upper triangle (ie every entry (i, j, value) has i <= j). Repeated
// coordinates are summed, so degrees may be appended as separate diagonal
// entries.
double compute_log_det_from_entries(std::vector<SparseEntry> const &entries,
                                    int const num_rows);



// Builds the minor of a weighted graph Laplacian one edge at a time and
// returns its log determinant (the log number of spanning trees).
//
// Small minors (dense_max_rows rows or fewer) are factored densely in a reused
// buffer; larger ones go through Eigen's sparse Cholesky, which lives behind
// SparseImpl in sparse_logdet.cpp so Eigen stays out of this header.
//
// Usage per Laplacian: reset(), add_edge() for each edge, then one log_det().
// Not thread safe; use one builder per thread or owner.
class LaplacianBuilder {
  public:
    static constexpr int dense_max_rows = 48;

    // preallocate_dense = false skips the dense buffer (dense_max_rows^2
    // doubles, about 18 KB) for owners that may never compute a log det. Such
    // a builder must not get a dense reset(); with perf_config::bounds_checking
    // on that throws, otherwise it writes past the empty buffer.
    explicit LaplacianBuilder(bool preallocate_dense = true);
    ~LaplacianBuilder();
    LaplacianBuilder(LaplacianBuilder &&) noexcept;
    LaplacianBuilder &operator=(LaplacianBuilder &&) noexcept;
    LaplacianBuilder(LaplacianBuilder const &) = delete;
    LaplacianBuilder &operator=(LaplacianBuilder const &) = delete;

    // Start a Laplacian on vertices 0, ..., num_vertices - 1 with row and
    // column `dropped_vertex` removed. num_vertices >= 1 (one vertex gives an
    // empty minor with log det 0) and 0 <= dropped_vertex < num_vertices.
    void reset(int num_vertices, int dropped_vertex = 0);

    // Undirected edge u-v with non-negative weight w (1 on the map graph, an
    // edge count on multigraphs). Adds w to both degrees and -w off the
    // diagonal; an endpoint equal to the dropped vertex only adds to the
    // other's degree. u == v is ignored.
    void add_edge(int u, int v, double w = 1.0);

    // Log determinant of the minor. Throws if it is not positive definite
    // (a disconnected graph). Call once per reset().
    double log_det();

  private:
    int num_vertices = 0;
    int dropped_vertex = 0;
    int num_rows = 0; // num_vertices - 1
    bool is_dense = true;
    // set by reset() and cleared by log_det(), since the dense factorization
    // overwrites its buffer and a second log_det() would read garbage
    bool awaiting_log_det = false;
    // dense_max_rows^2 when preallocated (else empty), column major, upper triangle used
    std::vector<double> dense_matrix;
    struct SparseImpl;                // Eigen triplets, degrees, SparseMatrix, SimplicialLLT
    std::unique_ptr<SparseImpl> sparse; // created on the first sparse reset()
};