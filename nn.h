#include"matrixutil.h"
#include<fstream>
#include<sstream>
#include<algorithm>
#include <cmath>
#include <stdexcept>

//adding new user
namespace nn {
    enum INIT_TYPE {
        ZEROS,
        ONES,
        RANDN,
        RAND,
        WEIGHT,
        HE
    };

    template<typename T>class NN {
        std::vector<size_t> units_per_layer;
        std::vector<alg::Matrix<T>> weights;
        std::vector<alg::Matrix<T>> biases;
        // std::vector<alg::Matrix<T>> new_weights;
        // std::vector<alg::Matrix<T>> new_biases;
        // std::vector<alg::Matrix<T>> activations;//a
        std::vector<alg::Matrix<T>> pre_activations;//Z
        std::string file_name;
        std::string line;
        std::ifstream file;
        std::ofstream w_file;

        float lr;

        INIT_TYPE initialization_type;
    public:
    std::vector<alg::Matrix<T>> activations;//a
        explicit NN(std::vector<size_t> units_per_layer, float lr=0.01f, std::string file_name="", INIT_TYPE initialization_type=INIT_TYPE::RAND) :
            units_per_layer(units_per_layer),
            weights(),
            biases(),
            activations(),
            lr(lr),
            file_name(file_name),
            initialization_type(initialization_type) {
                alg::Matrix<T> weight;
                alg::Matrix<T> bias;

                if(initialization_type==INIT_TYPE::WEIGHT) {
                    assert(file_name!="");
                    file = std::ifstream(file_name);
                }

                for(size_t u=0;u<units_per_layer.size()-1;++u) {
                    size_t input_layer_size = units_per_layer[u];
                    size_t output_layer_size = units_per_layer[u+1];
                
                    if(initialization_type==INIT_TYPE::ZEROS) {
                        weight = alg_util::Util<T>::zeros(output_layer_size, input_layer_size);
                        bias = alg_util::Util<T>::zeros(1, output_layer_size);
                    } else if(initialization_type==INIT_TYPE::ONES) {
                        weight = alg_util::Util<T>::ones(output_layer_size, input_layer_size);
                        bias = alg_util::Util<T>::ones(1, output_layer_size);
                    } else if(initialization_type==INIT_TYPE::RAND) {
                        weight = alg_util::Util<T>::rand(output_layer_size, input_layer_size);
                        bias = alg_util::Util<T>::rand(1, output_layer_size);
                    } else if(initialization_type==INIT_TYPE::RANDN) {
                        weight = alg_util::Util<T>::randn(output_layer_size, input_layer_size);
                        bias = alg_util::Util<T>::randn(1, output_layer_size);
                    } else if(initialization_type==INIT_TYPE::WEIGHT) {
                        //weight
                        assert(std::getline(file, line));
                        
                        size_t rows, cols;
                        std::string cell;
                        std::stringstream ss(line);
                        std::getline(ss, cell, ',');
                        rows = std::stoi(cell);
                        std::getline(ss, cell, ',');
                        cols = std::stoi(cell);
                    
                        assert(output_layer_size==rows);
                        assert(input_layer_size==cols);

                        std::vector<T> fill_vector;
                        while(std::getline(ss, cell, ',')) {
                            fill_vector.push_back(stof(cell));
                        }
                        weight = alg::Matrix<T>(output_layer_size, input_layer_size);
                        weight.fill_vector(fill_vector);
                        //bias
                        assert(std::getline(file, line));

                        ss = std::stringstream(line);
                        std::getline(ss, cell, ',');
                        rows = std::stoi(cell);
                        std::getline(ss, cell, ',');
                        cols = std::stoi(cell);
                        
                        assert(1==rows);
                        assert(output_layer_size==cols);

                        fill_vector = std::vector<T>();
                        while(std::getline(ss, cell, ',')) {
                            fill_vector.push_back(stof(cell));
                        }
                        bias = alg::Matrix<T>(1, output_layer_size);
                        bias.fill_vector(fill_vector);

                    } else if(initialization_type==INIT_TYPE::HE) {
                        weight = alg_util::Util<T>::he(output_layer_size, input_layer_size);
                        bias = alg_util::Util<T>::ones(1, output_layer_size);
                    }else {
                        std::cout<<"Invalid initialization_type value"<<std::endl;
                        assert(false);
                    }

                    weights.push_back(weight);
                    biases.push_back(bias);
                }

                if(initialization_type==INIT_TYPE::WEIGHT) {
                    file.close();
                }

                activations.resize(units_per_layer.size());
                pre_activations.resize(units_per_layer.size());
        }

        inline static T relu(T x) {
            return x > 0 ? x : static_cast<T>(0);
        }

        inline static T d_relu(T x) {
            return x > 0 ? static_cast<T>(1) : static_cast<T>(0);
        }

        inline static T sigmoid(T x) {
            return static_cast<T>(1) / (static_cast<T>(1) + std::exp(-x));
        }
        
        inline static T d_sigmoid(T x) {
            T s = sigmoid(x);
            return s * (static_cast<T>(1) - s);
        }

        inline static T leaky_relu(T x) {
            return x > 0 ? x : static_cast<T>(0.01) * x;
        }

        inline static T d_leaky_relu(T x) {
            return x > 0 ? static_cast<T>(1) : static_cast<T>(0.1);
        }

        inline static T custom_relu(T x) {
            return (x<=1e+12f && x>=-1e+12f) ? x : (x<0 ? -1e+12f : 1e+12f);
        }

        inline static T d_custom_relu(T x) {
            return (x<=1e+12f && x>=-1e+12f) ? 1 : 0;
        }

        inline static T clamp(T x, T eps = static_cast<T>(1e-12)) {
            return std::min(std::max(x, eps), static_cast<T>(1.0 - eps));
        }

        alg::Matrix<T> binary_cross_entropy(alg::Matrix<T> y_true, alg::Matrix<T> y_pred) {
            alg::Matrix<T> log_y_pred = y_pred.apply_function([](T x){return log(x);});
            alg::Matrix<T> one_y_pred = alg_util::Util<T>::ones(y_pred.dim[0], y_pred.dim[1])-y_pred;
            alg::Matrix<T> one_y_true = alg_util::Util<T>::ones(y_true.dim[0], y_true.dim[1])-y_true;
            alg::Matrix<T> log_one_y_pred = one_y_pred.apply_function([](T x){return log(x);});
            return (y_true.multiply_elementwise(log_y_pred)+one_y_true.multiply_elementwise(log_one_y_pred)).scalar_multiply(-1.0f);
        }

        alg::Matrix<T> round_off(alg::Matrix<T> m) {
            alg::Matrix<T> new_m(m.dim[0], m.dim[1]);
            for(size_t row=0;row<m.dim[0];++row) {
                for(size_t col=0;col<m.dim[1];++col) {
                    if(std::abs(m(row, col))<=1e-6f) {
                        new_m(row, col) = 1e-15f;
                    }
                }
            }

            return new_m;
        }
        
        alg::Matrix<T> binary_cross_entropy_derivative(alg::Matrix<T> y_true, alg::Matrix<T> y_pred) {
            std::cout<<"y_pred:"<<std::endl;
            y_pred.print();
            std::cout<<"y_true:"<<std::endl;
            y_true.print();
            y_pred = y_pred.apply_function([this](T x){return clamp(x);});
            alg::Matrix<T> div = y_true.divide_elementwise(y_pred);
            std::cout<<"div:"<<std::endl;
            div.print();
            alg::Matrix<T> one_y_true = alg_util::Util<T>::ones(y_true.dim[0], y_true.dim[1])-y_true;
            // one_y_true = round_off(one_y_true);
            std::cout<<"one_y_true:"<<std::endl;
            one_y_true.print();

            alg::Matrix<T> one_y_pred = alg_util::Util<T>::ones(y_pred.dim[0], y_pred.dim[1])-y_pred;
            // one_y_pred = round_off(one_y_pred);

            std::cout<<"one_y_pred:"<<std::endl;
            one_y_pred.print();
            alg::Matrix<T> one_div = one_y_true.divide_elementwise(one_y_pred);
            std::cout<<"one_div:"<<std::endl;
            one_div.print();

            return (div-one_div).scalar_multiply(-1.0f);
            // return - (y_true / y_pred - (1 - y_true) / (1 - y_pred));
        }

        alg::Matrix<T> forward(alg::Matrix<T> input) {
            activations[0] = input;
            pre_activations[0] = input;

            for(size_t layer=0;layer<units_per_layer.size()-1;++layer) {
                // pre_activations[layer+1] = (activations[layer]*(~weights[layer]))+biases[layer];
                // activations[layer+1] = pre_activations[layer+1].apply_function([this](T x){return relu(x);});
                if(layer==units_per_layer.size()-2) {
                    pre_activations[layer+1] = (activations[layer]*(~weights[layer]))+biases[layer];
                    activations[layer+1] = pre_activations[layer+1];
                } else {
                    pre_activations[layer+1] = (activations[layer]*(~weights[layer]))+biases[layer];
                    activations[layer+1] = pre_activations[layer+1].apply_function([this](T x){return relu(x);});
                }

            }

            return activations.back();

        }

        void backprop(alg::Matrix<T> target) {
            alg::Matrix<T> error = (activations.back()-target);
            // alg::Matrix<T> error = binary_cross_entropy_derivative(target, activations.back());
            // std::cout<<"error"<<std::endl;
            // error.print();
            // std::cout<<"error"<<std::endl;

            std::vector<alg::Matrix<T>> dW(weights.size());
            std::vector<alg::Matrix<T>> dB(weights.size());


            for(size_t layer=units_per_layer.size()-1;layer>0;--layer) {
                // std::cout<<"error"<<std::endl;
                // error.print();
                // std::cout<<"error"<<std::endl;
                // std::cout<<std::endl;


                alg::Matrix<T> z = pre_activations[layer];
                alg::Matrix<T> dZ;
                if(layer==units_per_layer.size()-1) {
                    // dZ = error.multiply_elementwise(z.apply_function([this](T x){return d_sigmoid(x);}));
                    dZ = error;
                } else {
                    dZ = error.multiply_elementwise(z.apply_function([this](T x){return d_relu(x);}));
                }
                // alg::Matrix<T> dZ = error.multiply_elementwise(z.apply_function([this](T x){return d_relu(x);}));//correct
                
                // weight and bias derivative
                dW[layer-1] = (~dZ)*activations[layer-1];//correct
                dB[layer-1] = dZ;//correct


                // alg::Matrix<T> weights_prev = weights[layer-1];

                // weight update
                
                // weights[layer-1] = (weights[layer-1]-(dW.scalar_multiply(lr)));
                // biases[layer-1] = (biases[layer-1]-(dB.scalar_multiply(lr)));


                error = (dZ*weights[layer-1]).norm();//correct
                // error = (dZ*weights_prev);

            }

            //weight update
            for(size_t layer=units_per_layer.size()-1;layer>0;--layer) {
                weights[layer-1] = (weights[layer-1]-(dW[layer-1].scalar_multiply(lr)));
                biases[layer-1] = (biases[layer-1]-(dB[layer-1].scalar_multiply(lr)));
            }
        }

        void backprop_(alg::Matrix<T> target, std::vector<alg::Matrix<T>> activations, std::vector<alg::Matrix<T>> pre_activations) {
            alg::Matrix<T> error = (activations.back()-target);
            // alg::Matrix<T> error = binary_cross_entropy_derivative(target, activations.back());
            // std::cout<<"error"<<std::endl;
            // error.print();
            // std::cout<<"error"<<std::endl;

            std::vector<alg::Matrix<T>> dW(weights.size());
            std::vector<alg::Matrix<T>> dB(weights.size());


            for(size_t layer=units_per_layer.size()-1;layer>0;--layer) {
                // std::cout<<"error"<<std::endl;
                // error.print();
                // std::cout<<"error"<<std::endl;
                // std::cout<<std::endl;


                alg::Matrix<T> z = pre_activations[layer];
                alg::Matrix<T> dZ;
                if(layer==units_per_layer.size()-1) {
                    // dZ = error.multiply_elementwise(z.apply_function([this](T x){return d_sigmoid(x);}));
                    dZ = error;
                } else {
                    dZ = error.multiply_elementwise(z.apply_function([this](T x){return d_relu(x);}));
                }
                // alg::Matrix<T> dZ = error.multiply_elementwise(z.apply_function([this](T x){return d_relu(x);}));//correct
                
                // weight and bias derivative
                dW[layer-1] = ((~dZ)*activations[layer-1]).norm();//correct
                dB[layer-1] = dZ.norm();//correct


                // alg::Matrix<T> weights_prev = weights[layer-1];

                // weight update
                
                // weights[layer-1] = (weights[layer-1]-(dW.scalar_multiply(lr)));
                // biases[layer-1] = (biases[layer-1]-(dB.scalar_multiply(lr)));


                error = (dZ*weights[layer-1]).norm();//correct
                // error = (dZ*weights_prev);

            }

            //weight update
            for(size_t layer=units_per_layer.size()-1;layer>0;--layer) {
                weights[layer-1] = (weights[layer-1]-(dW[layer-1].scalar_multiply(lr)));
                biases[layer-1] = (biases[layer-1]-(dB[layer-1].scalar_multiply(lr)));
            }
        }

        void print_weights() {
            for(alg::Matrix<T> matrix:weights) {
                matrix.print();
            }
        }

        void print_nodes() {
            for(size_t layer=0;layer<units_per_layer.size();++layer) {
                activations[layer].print();
            }
        }

        void save_weights() {
            assert(file_name!="");
            w_file = std::ofstream(file_name);
            
            for(size_t l=0;l<units_per_layer.size()-1;++l) {
                //save weights
                w_file<<weights[l].dim[0]<<','<<weights[l].dim[1];
                for(size_t r=0;r<weights[l].dim[0];++r) {
                    for(size_t c=0;c<weights[l].dim[1];++c) {
                        w_file<<','<<weights[l](r, c);
                    }
                }
                w_file<<std::endl;

                //save bias
                w_file<<biases[l].dim[0]<<','<<biases[l].dim[1];
                for(size_t r=0;r<biases[l].dim[0];++r) {
                    for(size_t c=0;c<biases[l].dim[1];++c) {
                        w_file<<','<<biases[l](r, c);
                    }
                }
                w_file<<std::endl;
            }

            w_file.close();

        }

        void snapshot(std::vector<alg::Matrix<T>> &activations_, std::vector<alg::Matrix<T>> &pre_activations_) {
            for(alg::Matrix<T> activation:activations) {
                activations_.push_back(activation);
            }
            for(alg::Matrix<T> pre_activation:pre_activations) {
                pre_activations_.push_back(pre_activation);
            }
        }
    };
}