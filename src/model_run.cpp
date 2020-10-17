
#include <chrono>
#include <iomanip>
#include "CatDogCNN.h"
using namespace std;
using namespace chrono;
using namespace tensorflow;
using namespace tensorflow::ops;

int main(int argc, const char * argv[])
{
    int image_side = 150;
    int image_channels = 3;
    CatDogCNN model(image_side, image_channels);
    Status s;
    std::string proto_name = "/media/aytac/Tank/machine-learning/TFcpp/graphs/model2.pb";
    s = model.LoadSavedModel(proto_name);
    TF_CHECK_OK(s);

    string base_folder; 
    //testing the model
    s = model.CreateGraphForImage(false);//rebuild the model without unstacking
    TF_CHECK_OK(s);
    base_folder = "/media/aytac/Tank/machine-learning/TFcpp/data/cats_and_dogs_small/cats_and_dogs_small/test";
    vector<pair<Tensor, float>> all_files_tensors;
    s = model.ReadFileTensors(base_folder, {make_pair("cats", 0), make_pair("dogs", 1)}, all_files_tensors);
    TF_CHECK_OK(s);
    //test the images
    int count_success = 0;
    for(int i = 0; i < all_files_tensors.size(); i++)
    {
        pair<Tensor, float> p = all_files_tensors[i];
        int result;
        s = model.PredictFromFrozen(p.first, result);
        TF_CHECK_OK(s);
        if(i%10 == 0)
            cout << "Test number: " << i + 1 << " predicted: " << result << " actual is: " << p.second << endl;
        if(result == (int)p.second)
            count_success++;
    }
    cout << "total successes: " << count_success << " out of " << all_files_tensors.size() << " which is " << setprecision(5) << (float)count_success / all_files_tensors.size() * 100 << "%" << endl;
    

    return 0;
}
