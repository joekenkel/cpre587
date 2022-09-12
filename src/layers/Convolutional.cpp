#include <iostream>

#include "../Utils.h"
#include "../Types.h"
#include "Layer.h"
#include "Convolutional.h"


namespace ML {
    // --- Begin Student Code ---

    // Compute the convultion for the layer data
    void ConvolutionalLayer::computeNaive(const LayerData &dataIn) const {
        // TODO: Your Code Here...

        bool debug = false;

        if (debug)
            std::cout << "\n\n\n";

        std::cout << getInputParams().dims[3];

        //Define Parameters
        int input_height = getInputParams().dims[0];
        int input_width = getInputParams().dims[1];
        int num_input_channels = getInputParams().dims[2];
        int filter_height = getWeightParams().dims[0];
        int filter_width = getWeightParams().dims[1];
        int num_filter_channels = getWeightParams().dims[3];
        int output_height, output_width;

        //Probably have a variable assicated with this        
        int batch_size = 1;
        int step_size = 1;

        output_height = ((input_height - filter_height + step_size) / step_size);
        output_width = ((input_width - filter_width + step_size) / step_size);

        //check
        if(debug){
            std::cout << "Input Height = " << input_height << ", Input Width = " << input_width << ", num_input_channels " << num_input_channels << "\n"
                    << "Filter Height = " << filter_height << ", Filter Width = " << filter_width << ", num_filter_channels " << num_filter_channels << "\n"
                    << "Output Height = " << output_height << ", Output Width = " << output_width << ", num_output_channels " << num_input_channels << "\n"
                    << "\n\n";
        }

        //Preset LayerDate Tpye
        LayerData Weight_data = getWeightData();
        LayerData Bias_data = getBiasData();
        LayerData Output_data = getOutputData();

        //Map values to memory
        Array4D_fp32 convWeightData = Weight_data.getData<Array4D_fp32>();
        Array1D_fp32 convBiasData = Bias_data.getData<Array1D_fp32>();
        Array3D_fp32 convInputData = dataIn.getData<Array3D_fp32>();
        Array3D_fp32 convOutputData = Output_data.getData<Array3D_fp32>();

        //predeclair variables
        int n,m,p,q,c,r,s;

        //Debugging Var - var[x][y][z]
        int input_x, input_y;


        for(n = 0; n < batch_size; n++){
            for(m = 0; m < num_filter_channels; m++){
                for(p = 0; p < output_height; p++){
                    for(q = 0; q < output_width; q++){
                       for(c = 0; c < num_input_channels; c++){
                            for(r = 0; r < filter_height; r++){
                                for(s = 0; s < filter_width; s++){
                                    
                                    input_x = step_size * q + s;
                                    input_y = step_size * p + r;

                                    convOutputData[q][p][m] = convInputData[input_x][input_y][c] * convWeightData[s][r][c][m];
                                }
                            }
                        } 
                        convOutputData[q][p][m] += convBiasData[m];
                    }
                } 
            }
        }
        std::cout << "\n\n\n";
    }


    // Compute the convolution using threads
    void ConvolutionalLayer::computeThreaded(const LayerData &dataIn) const {
        // TODO: Your Code Here...

        std::cout << "\n\n\nhello Convolution thread\n\n\n";

    }


    // Compute the convolution using a tiled approach
    void ConvolutionalLayer::computeTiled(const LayerData &dataIn) const {
        // TODO: Your Code Here...

        std::cout << "\n\n\nhello Convolution Tiled\n\n\n";

    }


    // Compute the convolution using SIMD
    void ConvolutionalLayer::computeSIMD(const LayerData &dataIn) const {
        // TODO: Your Code Here...

        std::cout << "\n\n\nhello Convolution SIMD\n\n\n";

    }
};