// #include"matrix.h"
// #include"matrixutil.h"
#include"nn.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

void train_and_test();

int main() {
    // alg::Matrix<int> m = alg_util::Util<int>::zeros(2, 2);
    // alg::Matrix<int> m1 = alg_util::Util<int>::ones(2, 2);
    // // alg::Matrix<float> m2 = alg_util::Util<float>::rand(2, 2);
    // // alg::Matrix<float> m3 = alg_util::Util<float>::randn(2, 2);
    // std::vector<int> v = {1, 2, 3, 4};

    // m.fill_vector(v);
    // m1.fill_vector(v);
    
    // m.print();
    // m1.print();
    // m2.print();
    // m3.print();

    // (m+m1).print();
    // (m-m1).print();

    // std::vector<size_t> units_per_layer = {3, 3, 3};
    // nn::NN<float> model = nn::NN<float>(units_per_layer, 0.01f, nn::INIT_TYPE::ONES);

    // alg::Matrix<float> input(1, 3);
    // input.fill_vector({1.0, 2.0, 3.0});

    // (model.forward(input)).print();

    // (m.multiply_elementwise(m1)).print();

    train_and_test();
}

void train_and_test() {
    std::vector<alg::Matrix<float>> x;
    std::vector<alg::Matrix<float>> y;

    std::ifstream file("Housing.csv");

    if (!file.is_open()) {
        std::cerr << "Failed to open file: "  << std::endl;
        return;
    }

    std::string line;
    std::getline(file, line);
    std::cout<<line<<std::endl;
    while (std::getline(file, line)) {
        std::vector<float> row;
        std::stringstream ss(line);
        std::string cell;

        alg::Matrix<float> y_temp(1, 1);
        std::vector<float> y_temp_fill;

        std::getline(ss, cell, ',');

        y_temp_fill.push_back(std::stof(cell)/10000);
        y_temp.fill_vector(y_temp_fill);
        // y_temp.print();
        y.push_back(y_temp);

        while (std::getline(ss, cell, ',')) {
            // std::cout<<cell<<std::endl;
            if(cell.compare("yes")==0) {
                row.push_back(1);
            } else if(cell.compare("no")==0) {
                row.push_back(0);
            } else if(cell.compare("unfurnished\r")==0) {
                row.push_back(0);
            } else if(cell.compare("semi-furnished\r")==0) {
                row.push_back(1.0);
            } else if(cell.compare("furnished\r")==0) {
                row.push_back(2.0);
            } else {
                
                row.push_back(std::stof(cell));
            }
        }
        // std::cout<<row.size()<<std::endl;
        alg::Matrix<float> x_temp(1, 12);
        x_temp.fill_vector(row);
        // x_temp.print();
        x.push_back(x_temp);


    }

    file.close();

    std::cout<<y.size()<<std::endl;
    std::cout<<x.size()<<std::endl;
    std::vector<size_t> layers = {12, 32, 1};
    nn::NN<float> model = nn::NN<float>(layers, 0.00001f, "", nn::INIT_TYPE::RANDN);
    // model.print_weights();
    for(int ep=0;ep<5000;++ep) {
    for(int i=0;i<400;i++) {
        // x[i].print();
        // y[i].print();
        model.forward(x[i]);
        model.backprop(y[i]);
        // model.print_weights();
    }}

    model.print_weights();

    alg::Matrix<float> mse = alg_util::Util<float>::zeros(1, 1);
    for(int i=400;i<500;i++) {
        alg::Matrix<float> pred = model.forward(x[i]);
        mse = mse + (pred-y[i]).apply_function([](int x){return pow(x, 2);});
    }

    mse.scalar_multiply(0.01);
    mse.print();

    // model.print_weights();
    x[423].print();
    x[541].print();
    x[544].print();
    model.forward(x[423]).print();
    model.forward(x[541]).print();
    model.forward(x[544]).print();

}