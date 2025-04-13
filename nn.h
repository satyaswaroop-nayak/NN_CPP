#include"matrixutil.h"

namespace nn {
    enum INIT_TYPE {
        ZEROS,
        ONES,
        RANDN,
        RAND
    };

    template<typename T>class NN {
        std::vector<size_t> units_per_layer;
        std::vector<alg::Matrix<T>> weights;
        std::vector<alg::Matrix<T>> biases;
        std::vector<alg::Matrix<T>> activations;//a
        std::vector<alg::Matrix<T>> pre_activations;//Z

        float lr;

        INIT_TYPE initialization_type;
    public:
        explicit NN(std::vector<size_t> units_per_layer, float lr=0.01f, INIT_TYPE initialization_type=INIT_TYPE::RAND) :
            units_per_layer(units_per_layer),
            weights(),
            biases(),
            activations(),
            lr(lr),
            initialization_type(initialization_type) {
                alg::Matrix<T> weight;
                alg::Matrix<T> bias;
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
                    } else {
                        std::cout<<"Invalid initialization_type value"<<std::endl;
                        assert(false);
                    }

                    weights.push_back(weight);
                    biases.push_back(bias);
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

        alg::Matrix<T> forward(alg::Matrix<T> input) {
            activations[0] = input;
            pre_activations[0] = input;

            for(size_t layer=0;layer<units_per_layer.size()-1;++layer) {
                pre_activations[layer+1] = (activations[layer]*(~weights[layer]))+biases[layer];
                activations[layer+1] = pre_activations[layer+1].apply_function([this](T x){return relu(x);});

            }

            return activations[units_per_layer.size()-1];

        }

        void backprop(alg::Matrix<T> target) {
            alg::Matrix<T> error = activations.back()-target;
            std::cout<<"error"<<std::endl;
            error.print();
            std::cout<<"error"<<std::endl;

            for(size_t layer=units_per_layer.size()-1;layer>0;--layer) {
                alg::Matrix<T> z = pre_activations[layer];
                alg::Matrix<T> dZ = error.multiply_elementwise(z.apply_function([this](T x){return d_relu(x);}));

                std::cout<<"dZ"<<std::endl;
                dZ.print();
                std::cout<<"dZ"<<std::endl;
                
                // weight and bias derivative
                alg::Matrix<T> dW = (~dZ)*activations[layer-1];
                alg::Matrix<T> dB = dZ;

                // weight update
                weights[layer-1] = weights[layer-1]-(dW.scalar_multiply(lr));
                biases[layer-1] = biases[layer-1]-(dB.scalar_multiply(lr));



                error = (dZ*weights[layer-1]).multiply_elementwise(pre_activations[layer-1].apply_function([this](T x){return d_relu(x);}));

            }
        }

        void print_weights() {
            for(alg::Matrix<T> matrix:weights) {
                matrix.print();
            }
        }
    };
}