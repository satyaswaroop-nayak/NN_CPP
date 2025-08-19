#include"matrix.h"
#include <random>

namespace alg_util {
    template<typename T>class Util {
    public:
        static alg::Matrix<T> zeros(size_t rows, size_t cols) {
            alg::Matrix<T> zero_matrix(rows, cols);
            for(size_t r=0;r<rows;++r) {
                for(size_t c=0;c<cols;++c) {
                    zero_matrix(r, c) = 0;
                }
            }

            return zero_matrix;
        }

        static alg::Matrix<T> ones(size_t rows, size_t cols) {
            alg::Matrix<T> one_matrix(rows, cols);
            for(size_t r=0;r<rows;++r) {
                for(size_t c=0;c<cols;++c) {
                    one_matrix(r, c) = 1;
                }
            }

            return one_matrix;
        }

        static alg::Matrix<T> randn(size_t rows, size_t cols) {
            alg::Matrix<T> M{rows, cols};
        
            std::random_device rd{};
            std::mt19937 gen{rd()};
            T n(M.numel);
            T stdev{1 / sqrt(n)};
            std::normal_distribution<T> d{0, stdev};
        
            for (size_t r = 0; r < rows; ++r) {
              for (int c = 0; c < cols; ++c) {
                M(r, c) = d(gen);
              }
            }
            return M;
          }

        static alg::Matrix<T> rand(size_t rows, size_t cols) {
            alg::Matrix<T> M{rows, cols};
        
            std::random_device rd{};
            std::mt19937 gen{rd()};
            std::uniform_real_distribution<T> d{0, 1};
        
            for (size_t r = 0; r < rows; ++r) {
              for (int c = 0; c < cols; ++c) {
                M(r, c) = d(gen);
              }
            }
            return M;
          }

          static alg::Matrix<T> he(size_t rows, size_t cols) {
            alg::Matrix<T> M{rows, cols};
        
            double limit = sqrt(2.0 / cols);  // He uniform

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> d(-limit, limit);
        
            for (size_t r = 0; r < rows; ++r) {
              for (int c = 0; c < cols; ++c) {
                M(r, c) = d(gen);
              }
            }
            return M;
          }
        
    };
}