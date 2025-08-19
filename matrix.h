#include<vector>
#include<cassert>
#include<iostream>
#include<functional>

namespace alg {
template<typename T>class Matrix {
    size_t rows, cols; //0 indexed
    std::vector<T> elements;
public:
    size_t numel;
    std::vector<size_t> dim;
    explicit Matrix(size_t rows=0, size_t cols=0) : rows(rows), cols(cols) {
        elements.resize(rows*cols, T());
        numel = rows*cols;
        dim = {rows, cols};
    }

    void fill_vector(std::vector<T> fill_data) {
        assert(elements.size() == fill_data.size());
        for(size_t i=0;i<elements.size();++i) {
            elements[i] = fill_data[i];
        }
    }

    T& operator()(size_t row, size_t col) {
        assert (0<=row && row<rows);
        assert (0<=col && col<cols);
        return elements[row*cols + col];
    }

    // Transpose of a matrix
    Matrix<T> operator~() {
        Matrix<T> new_m = Matrix(cols, rows);
        if(cols == 1 || rows == 1) {
            new_m.fill_vector(elements);
        } else {
            for(size_t r=0;r<rows;++r) {
                for(size_t c=0;c<cols;++c) {
                    new_m(c, r) = (*this)(r, c);
                }
            }
        }
        return new_m;
    }

    // Matrix multiplication
    Matrix<T> operator*(Matrix<T> b) {
        assert((*this).cols == b.rows);
        Matrix<T> mult = Matrix((*this).rows, b.cols);
        for(size_t r=0;r<(*this).rows;++r) {
            for(size_t c=0;c<b.cols;++c) {
                mult(r, c) = 0;
                for(size_t e=0;e<(*this).cols;++e) {
                    mult(r, c) = mult(r, c) + ((*this)(r, e)*b(e, c));
                }
            }
        }

        return mult;
    }

    // Matrix addition
    Matrix<T> operator+(Matrix<T> b) {
        assert((*this).rows == b.rows);
        assert((*this).cols == b.cols);
        Matrix<T> add = Matrix((*this).rows, (*this).cols);
        for(size_t e=0;e<numel;++e) {
            add.elements[e] = (*this).elements[e]+b.elements[e];
        }
        return add;
    }

    // Matrix subtraction
    Matrix<T> operator-(Matrix<T> b) {
        assert((*this).rows == b.rows);
        assert((*this).cols == b.cols);
        Matrix<T> sub = Matrix((*this).rows, (*this).cols);
        for(size_t e=0;e<numel;++e) {
            sub.elements[e] = (*this).elements[e]-b.elements[e];
            // if(std::abs(sub.elements[e])<=1e-6f) {
            //     sub.elements[e] = 0;
            // }
        }
        return sub;
    }

    Matrix<T> multiply_elementwise(Matrix<T> b) {
        assert((*this).rows == b.rows);
        assert((*this).cols == b.cols);
        Matrix<T> res = Matrix((*this).rows, (*this).cols);
        for(size_t e=0;e<numel;++e) {
            res.elements[e] = (*this).elements[e]*b.elements[e];
        }
        return res;
    }

    Matrix<T> divide_elementwise(Matrix<T> b) {
        assert((*this).rows == b.rows);
        assert((*this).cols == b.cols);
        Matrix<T> res = Matrix((*this).rows, (*this).cols);
        for(size_t e=0;e<numel;++e) {
            res.elements[e] = (*this).elements[e]/b.elements[e];
        }
        return res;
    }

    Matrix<T> scalar_multiply(T x) {
        Matrix<T> res = Matrix((*this).rows, (*this).cols);
        for(size_t e=0;e<numel;++e) {
            res.elements[e] = (*this).elements[e]*x;
        }
        return res;
    }

    Matrix<T> apply_function(std::function<T(T)> function) {
        Matrix<T> new_m = Matrix<T>((*this).rows, (*this).cols);
        for(size_t e=0;e<numel;++e) {
            new_m.elements[e] = function((*this).elements[e]);
        }
        // for(size_t row=0;row<rows;++row) {
        //     for(size_t col=0;col<cols;++col) {
        //         new_m(row, col) = function((*this)(row, col));
        //     }
        // }

        return new_m;
    }

    Matrix<T> norm() {
        Matrix<T> new_m = Matrix<T>((*this).rows, (*this).cols);
        for(size_t e=0;e<numel;++e) {
            if((*this).elements[e]>=5.0f) {
                new_m.elements[e] = 5.0f;
            } else if((*this).elements[e]<=(-5.0f)) {
                new_m.elements[e] = -5.0f;
            } else if((*this).elements[e]<=1e-6f && (*this).elements[e]>=0) {
                new_m.elements[e] = 1e-6f;
            } else if((*this).elements[e]>=-1e-6f && (*this).elements[e]<=0) {
                new_m.elements[e] = -1e-6f;
            }
            else {
                new_m.elements[e] = (*this).elements[e];
            }
        }

        return new_m;
    }

    Matrix<T> squeeze() {
        assert((*this).rows==1 || (*this).cols==1);
        Matrix<T> new_m(1, 1);
        
        if((*this).rows==1) {
            for(size_t col=0;col<(*this).cols;++col) {
                new_m(0, 0) = new_m(0, 0)+(*this)(0, col);
            }
        } else {
            for(size_t row=0;row<(*this).rows;++row) {
                new_m(0, 0) = new_m(0, 0)+(*this)(row, 0);
            }
        }

        return new_m;
    }

    void print() {
        for(size_t r=0;r<rows;++r) {
            for(size_t c=0;c<cols;++c) {
                std::cout << (*this)(r, c) << " ";
            }
            std::cout<<std::endl;
        }
    }
};

}