#include <iostream>

#include "../Utils.h"
#include "../Types.h"
#include "Layer.h"
#include "Dense.h"


namespace ML {
    // --- Begin Student Code ---

    // Compute the full connected for the layer data
    void DenseLayer::computeNaive(const LayerData &dataIn) const {
        // TODO: Your Code Here...
       
        //Define Parameters
        int input_height = getInputParams().dims[0];
        int num_input_channels = getWeightParams().dims[1];

        //Probably have a variable assicated with this        
        int batch_size = 1;

        //Preset LayerDate Tpye
        LayerData Weight_data = getWeightData();
        LayerData Bias_data = getBiasData();
        LayerData Output_data = getOutputData();

        //Map values to memory
        Array2D_fp32 convWeightData = Weight_data.getData<Array2D_fp32>();
        Array1D_fp32 convBiasData = Bias_data.getData<Array1D_fp32>();
        Array1D_fp32 convInputData = dataIn.getData<Array1D_fp32>();
        Array1D_fp32 convOutputData = Output_data.getData<Array1D_fp32>();

        //predeclair variables
        int n,m,h;

        //loop through and perform the opperation
        for(n = 0; n < batch_size; n++){
            for(m = 0; m < num_input_channels; m++){
                for(h = 0; h < input_height; h++){                                    
                    convOutputData[m] = convInputData[h] *convWeightData[h][m];                                
                }
                convOutputData[m] += convBiasData[m];
            } 
        }
    }
    


    // Compute the convolution using threads
    void DenseLayer::computeThreaded(const LayerData &dataIn) const {
        // TODO: Your Code Here...

        std::cout << "\n\n\nhello Convolution thread\n\n\n";

    }

    // Compute the convolution using a tiled approach
    void DenseLayer::computeTiled(const LayerData &dataIn) const {
        // TODO: Your Code Here...

        std::cout << "\n\n\nhello Convolution Tiled\n\n\n";

    }


    // Compute the convolution using SIMD
    void DenseLayer::computeSIMD(const LayerData &dataIn) const {
        // TODO: Your Code Here...

        std::cout << "\n\n\nhello Convolution SIMD\n\n\n";

    }
};