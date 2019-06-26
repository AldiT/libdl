//
// Created by Aldi Topalli on 2019-06-18.
//

#include "data_handler.h"
#include "TensorWrapper.h"
#include <cstring>


data_handler::data_handler()
{
    data_array = new std::vector<data*>;
    test_data  = new std::vector<data*>;
    training_data = new std::vector<data*>;
    validation_data = new std::vector<data*>;
}

data_handler::~data_handler()
{
    //FREE all the used memory later
}

void data_handler::read_feature_vector(std::string path)
{
    uint32_t header[4]; //MAGIC | NUM_OF_IMAGES | ROW_SIZE | COL_SIZE
    unsigned char bytes[4];
    FILE *f = fopen(path.c_str(), "r");
    if(f)
    {
        for(int i = 0; i < 4; i++)
        {
            if(fread(bytes, sizeof(bytes), 1, f))
            {
                header[i] = convert_to_little_endian(bytes);
            }
        }

        printf("Done getting file header.\n");
        int image_size = header[2]*header[3];

        for(int i = 0; i < header[1]; i++){
            data *d = new data();
            uint8_t element[1];

            for(int j = 0; j < image_size; j++)
            {
                if(fread(element, sizeof(element), 1, f))
                {
                    d->append_to_feature_vector(element[0]);
                }else
                {
                    printf("Error reading from file.\n");
                    exit(-1);
                }
            }
            data_array->push_back(d);
        }
        printf("Successfully read and stored %lu featured vectors.\n", data_array->size());
    }else
    {
        printf("Could not find file.\n");
        exit(1);
    }
}

void data_handler::read_feature_label(std::string path)
{
    uint32_t header[2]; //MAGIC | NUM_OF_IMAGES
    unsigned char bytes[4];
    FILE *f = fopen(path.c_str(), "r");
    if(f)
    {
        for(int i = 0; i < 2; i++)
        {
            if(fread(bytes, sizeof(bytes), 1, f))
            {
                header[i] = convert_to_little_endian(bytes);
            }
        }

        printf("Done getting label file header.\n");

        for(int i = 0; i < header[1]; i++){
            uint8_t element[1];
            if(fread(element, sizeof(element), 1, f))
            {
                data_array->at(i)->set_label(element[0]);
            }else
            {
                printf("Error reading from file.\n");
                exit(-1);
            }
        }
        printf("Successfully read and stored labels\n");
    }else
    {
        printf("Could not find file.\n");
        exit(1);
    }
}

void data_handler::split_data()
{
    std::unordered_set<int> used_indexes;
    int train_size = data_array->size() * TRAIN_SET_PERCENT;
    int test_size  = data_array->size() * TEST_SET_PERCENT;
    int valid_size = data_array->size() * VALIDATION_SET_PERCENT;

    //Training data
    int count = 0;
    while(count < train_size)
    {
        int rand_index = rand() % data_array->size(); // 0 & data_array->size() - 1
        if(used_indexes.find(rand_index) == used_indexes.end()){
            training_data->push_back(data_array->at(rand_index));
            used_indexes.insert(rand_index);
            count++;
        }
    }


    //Test data
    count = 0;
    while(count < test_size)
    {
        int rand_index = rand() % data_array->size(); // 0 & data_array->size() - 1
        if(used_indexes.find(rand_index) == used_indexes.end()){
            test_data->push_back(data_array->at(rand_index));
            used_indexes.insert(rand_index);
            count++;
        }
    }

    //Valid data
    count = 0;
    while(count < valid_size)
    {
        int rand_index = rand() % data_array->size(); // 0 & data_array->size() - 1
        if(used_indexes.find(rand_index) == used_indexes.end()){
            validation_data->push_back(data_array->at(rand_index));
            used_indexes.insert(rand_index);
            count++;
        }
    }

    printf("Training data size: %lu.\n", training_data->size());
    printf("Test data size: %lu.\n", test_data->size());
    printf("Validation data size: %lu.\n", validation_data->size());
}

void data_handler::count_classes()
{
    int count = 0;
    for(unsigned i = 0; i < data_array->size(); i++){
        if(class_map.find(data_array->at(i)->get_label()) == class_map.end())
        {
            class_map[data_array->at(i)->get_label()] = count;
            data_array->at(i)->set_enumerated_label(count);
            count++;
        }
    }
    num_classes = count;
    printf("Successfully counted the number of the classes: %d.\n", num_classes);
}
uint32_t data_handler::convert_to_little_endian(const unsigned char* bytes)
{
    return (uint32_t) ((bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | (bytes[3]));
}


std::vector<data*> *data_handler::get_training_data()
{
    return training_data;
}
std::vector<data*> *data_handler::get_test_data()
{
    return test_data;
}
std::vector<data*> *data_handler::get_validation_data()
{
    return validation_data;
}

void data_handler::print_instance(int i){
    for(int j = 0; j < training_data->at(i)->get_feature_vector()->size(); j++){
        if(j % 28 == 0)
            printf("\n");
        printf(" %d ", training_data->at(i)->get_feature_vector()->at(j));
    }
}

libdl::TensorWrapper_Exp data_handler::convert_training_data_to_Eigen()
{
    int batch_size = training_data->size();
    int features = training_data->at(0)->get_feature_vector()->size();

    libdl::TensorWrapper_Exp result(batch_size, features);

    Matrix8u tmp(training_data->size(), training_data->at(0)->get_feature_vector()->size());


    for(int i = 0; i < batch_size; i++){
        for(int j = 0; j < features; j++){
            std::memcpy(&tmp(i, j), &training_data->at(i)->get_feature_vector()->at(j), sizeof(uint8_t));
        }
    }

    result.set_tensor(tmp.cast<double>(), 28, 28, 1);

    return result;
}

libdl::TensorWrapper_Exp data_handler::convert_training_labels_to_Eigen()
{
    int batch_size = training_data->size();

    libdl::TensorWrapper_Exp result(batch_size, 1);

    Matrix8u tmp(training_data->size(), 1);

    for(int i = 0; i < batch_size; i++){
        tmp(i, 0) = training_data->at(i)->get_label();
    }

    result.set_tensor(tmp.cast<double>(), 1, 1 , 1);

    return result;
}

Eigen::MatrixXd data_handler::normalize_data()
{
    Eigen::MatrixXd data = this->convert_training_data_to_Eigen().get_tensor();

    for(int instance = 0; instance < data.rows(); instance++){
        double mean = data.block(instance, 0, 1, 28*28).mean();
        data.block(instance, 0, 1, 28*28).unaryExpr([mean](double e)
        {
           return e-mean;
        });
    }

    return data;
}

//Testing main

/*
#include <iostream>

int main(){
    data_handler *dh = new data_handler();
    dh->read_feature_vector("../data/train-images-idx3-ubyte");
    dh->read_feature_label("../data/train-labels-idx1-ubyte");
    dh->split_data();
    dh->count_classes();

    libdl::TensorWrapper_Exp train_data   = dh->convert_training_data_to_Eigen();
    libdl::TensorWrapper_Exp train_labels = dh->convert_training_labels_to_Eigen();


    for(int i = 0; i <784; i++){
        std::cout << train_data.get_tensor().block(0, 0, 1, 784)(0, i) << " ";
        if(i % 28 == 0)
            std::cout << std::endl;
    }

    dh->print_instance(0);
    std::cout << "\nLabel: " << train_labels.get_tensor()(0, 0);
}*/