/*#ifdef NDEBUG
#undef NDEBUG
#endif*/

#include <limits>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <vector>
#include <iostream>
#include <time.h>
#include <random>

#include "train.h"

// #define COMPUTE_OBJECTIVE
// #define USE_ADAGRAD

double expit(double x) {
    if (x > 0) {
        return 1. / (1. + exp(-x));
    } else {
        return 1. - 1. / (1. + exp(x));
    }
}

double log1exp(double x) {
    if (x > 0) {
        return x + log1p(std::exp(-x));
    } else {
        return log1p(std::exp(x));
    }
}

double _flm_obj(size_t n_S, const long *S, size_t n, size_t dim_rep,
        double n_logz,
        double *utilities, double *W_rep) {
#define IDX_REP(w, v) ((w) * dim_rep + (v))
    double value = n_logz;
    int el;

    // utilities
    for (size_t i = 0; i < n_S; i++) {
        el = S[i];
        value += utilities[el];
    }

    double max;
    double w;

    // repulse facloc
    for (size_t d = 0; d < dim_rep; d++) {
        max = 0.0;
        for (size_t i = 0; i < n_S; i++) {
            el = S[i];
            w = W_rep[IDX_REP(el, d)];
            if (max < w) {
                max = w;
            }
            value -= w; // TODO
        }
        value += max;
    }

    return value;
}

// TODO Improve this
double logaddexp(double a, double b) {
    return std::log(std::exp(a) + std::exp(b));
}


// Parameters
// ==========
// data : the data samples separated by -1, first entry is the label 1 / 0
// data_size : the size of the above vector
// n_steps : how many SGD steps to perform
// eta_0, power, start_step : the step size used is
//
//      eta_0 / (start_step + iteration) ** power;
//
// weights : the weights will be stored here. Will not be initialized and the
//           provided data will be used as the first iterate. Assumed to be
//           stored in *column-first* order (FORTRAN). Should be of size n x m.
// unaries : the unaries will be stored here. Should be of size n
void train(const long *data, size_t data_size, long n_steps,
           double eta_0, double power, int start_step,
           const double *unaries_noise,
           double *weights, double *unaries, double *n_logz,
           size_t n, size_t m) {
#define IDX(w, v) ((w) * m + (v))
    double additive_noise = 1e-3;

    std::vector<long> start_indices;
    start_indices.reserve(data_size);
    start_indices.push_back(0);
    long n_noise = (data[0] == 0.);
    long n_model = (data[0] == 1.);
    for (size_t i = 1; i < data_size; i++) {
        if (data[i] == -1.) {
            assert(i + 1 < data_size);
            if (data[i+1] == 1.) {
                n_model++;
            } else {
                assert(data[i+1] == 0.);
                n_noise++;
            }
            start_indices.push_back(i + 1);
        }
    }

    double logz_noise = 0.;
    for (size_t i = 0; i < n; i++) {
        logz_noise += log1exp(unaries_noise[i]);
    }

    double log_nu = std::log(
        static_cast<double>(n_noise) / static_cast<double>(n_model));

    clock_t t;
    t = clock();

    // for adagrad
    double g_utilities[n];
    double g_W[n * m];
    double g_nlogz = 0;
    for (size_t i = 0; i < n; i++) {
        g_utilities[i] = 1e-2;
    }
    for (size_t i = 0; i < m * n; i++) {
        g_W[i] = 1e-2;
    }

    std::vector<int> perm(start_indices.size());
    for (size_t i = 0; i < start_indices.size(); i++) {
        perm[i] = i;
    }

    std::srand(start_step);
    size_t positions[m];  // indices of maxima
    for (int i = 0; i < n_steps; i++) {
        if (i % start_indices.size() == 0) {
            // std::random_shuffle(perm.begin(), perm.end());
#ifdef COMPUTE_OBJECTIVE
            double objective = 0;
            for (size_t sidx = 0; sidx < start_indices.size(); sidx++) {
                size_t start_idx = start_indices[sidx];
                size_t end_idx;
                if (sidx + 1 == start_indices.size()) {
                    end_idx = data_size;
                } else {
                    end_idx = start_indices[sidx + 1] - 1;
                }

                size_t len_S = end_idx - start_idx - 1;
                double tlabel = data[start_idx];

                double c_f_model = _flm_obj(len_S, &data[start_idx],
                    n, m, n_logz[0], unaries, weights);

                double c_f_noise = -logz_noise;
                for (size_t j = 0; j < len_S; j++) {
                    c_f_noise += unaries_noise[data[start_idx + j]];
                }

                double G = c_f_model - c_f_noise;
                double mul = tlabel > .5 ? 1. : -1.;
                objective -= logaddexp(0, mul * (log_nu - G));
            }
            std::cout << "Objective: " << objective << std::endl;
#endif
        }
        // size_t idx = perm[i % start_indices.size()];

        double random = static_cast<double>(std::rand()) / RAND_MAX;
        size_t idx = static_cast<size_t>(random * (start_indices.size() - 1));

        assert (idx < start_indices.size());
        size_t start_idx = start_indices[idx];
        size_t end_idx;
        if (idx + 1 == start_indices.size()) {
            end_idx = data_size;
        } else {
            end_idx = start_indices[idx + 1] - 1;
        }

        size_t len_S = end_idx - start_idx - 1;

        double f_model = *n_logz;
        double f_noise = -logz_noise;

        for (size_t k = start_idx + 1; k < end_idx; k++) {
            f_model += unaries[data[k]];
            f_noise += unaries_noise[data[k]];
        }

        for (size_t j = 0; j < m; j++) {
            double max = -std::numeric_limits<double>::infinity();
            int max_idx = -1;
            for (size_t k = start_idx + 1; k < end_idx; k++) {
                assert(data[k] >= 0);
                assert(IDX(data[k], j) >= 0);
                assert(IDX(data[k], j) < n * m);
                f_model -= weights[IDX(data[k], j)];
                if (weights[IDX(data[k], j)] > max) {
                    max = weights[IDX(data[k], j)];
                    max_idx = IDX(data[k], j);
                }
            }
            if (len_S == 0)
                break;
            assert(max_idx != -1);
            f_model += max;
            positions[j] = max_idx;
        }

        // We can now take the gradient step.
        double label = data[start_idx];
        double factor = (label - expit(f_model - f_noise - log_nu));
#ifdef USE_ADAGRAD
        double step = eta_0 * factor;
#else
        double step = eta_0 * factor / std::pow(static_cast<double>(i + 1), power);
#endif
        double factor_sq = factor * factor;

        for (size_t k = start_idx + 1; k < end_idx; k++) {
            assert(data[k] >= 0);
            assert(data[k] < n);
#ifdef USE_ADAGRAD
            g_utilities[data[k]] += factor_sq;
            unaries[data[k]] += step / std::sqrt(g_utilities[data[k]]);
#else
            unaries[data[k]] += step;
#endif
        }

        for (size_t j = 0; j < m; j++) {
            if (len_S == 0)
                break;

            for (size_t k = start_idx + 1; k < end_idx; k++) {
                assert(IDX(data[k], j) >= 0);
                assert(IDX(data[k], j) < n * m);
                if (positions[j] == IDX(data[k], j))
                    continue;
#ifdef USE_ADAGRAD
                g_W[IDX(data[k], j)] += factor_sq;
                weights[IDX(data[k], j)] -= step / std::sqrt(g_W[IDX(data[k], j)]);
#else
                weights[IDX(data[k], j)] -= step;
#endif
                if (weights[IDX(data[k], j)] <= 0) {
                     weights[IDX(data[k], j)] = additive_noise * (
                        static_cast<double>(std::rand()) / RAND_MAX);
                }
            }
        }

#ifdef USE_ADAGRAD
        g_nlogz += factor_sq;
        *n_logz += step / std::sqrt(g_nlogz);
#else
        *n_logz += step;
#endif
    }

    t = clock() - t;
    std::cout << "It took me "  << (((float)t)/CLOCKS_PER_SEC) << "seconds" << std::endl;
}

