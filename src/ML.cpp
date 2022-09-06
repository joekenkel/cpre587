#include <iostream>
#include <filesystem>
#include <vector>

#include "Utils.h"
#include "Types.h"
#include "Model.h"
#include "layers/Layer.h"
#include "layers/Convolutional.h"
#include "layers/Dense.h"
#include "layers/MaxPooling.h"
#include "layers/Softmax.h"

// Make our code a bit cleaner
namespace fs = std::filesystem;
using namespace ML;

// Build our ML toy model
Model buildToyModel(const fs::path modelPath) {
    Model model;
    std::cout << "\n--- Building Toy Model ---" << std::endl;

    // --- Conv 0: L0 ---
    // Input shape: 64x64x3
    // Output shape: 60x60x32
    // LayerParams conv1_inDataParam = getParams<fp32, "", 64, 64, 3>();
    // LayerParams conv1_outDataParam = getParams<fp32, "", 60, 60, 32>();
    // LayerParams conv1_weightParam = getParams<fp32, "", 5, 5, 3, 32>();
    // LayerParams conv1_biasParam = getParams<fp32, "", 32>();
    LayerParams conv1_inDataParam(sizeof(fp32), {64, 64, 3});
    LayerParams conv1_outDataParam(sizeof(fp32), {60, 60, 32});
    LayerParams conv1_weightParam(sizeof(fp32), {5, 5, 3, 32}, modelPath / "conv1_weights.bin");
    LayerParams conv1_biasParam(sizeof(fp32), {32}, modelPath / "conv1_biases.bin");

    ConvolutionalLayer* conv1 = new ConvolutionalLayer(conv1_inDataParam, conv1_outDataParam, conv1_weightParam, conv1_biasParam);
    model.addLayer(conv1);

    // --- Conv 1: L1 ---
    // Input shape: 60x60x32
    // Output shape: 56x56x32


    // --- MPL 0: L2 ---
    // Input shape: 56x56x32
    // Output shape: 28x28x32


    // --- Conv 2: L3 ---
    // Input shape: 28x28x32
    // Output shape: 26x26x64


    // --- Conv 3: L4 ---
    // Input shape: 26x26x64
    // Output shape: 24x24x64


    // --- MPL 1: L5---
    // Input shape: 24x24x64
    // Output shape: 12x12x64


    // --- Conv 4: L6 ---
    // Input shape: 12x12x64
    // Output shape: 10x10x64


    // --- Conv 5: L7 ---
    // Input shape: 10x10x64
    // Output shape: 8x8x128


    // --- MPL 2: L8 ---
    // Input shape: 8x8x128
    // Output shape: 4x4x128


    // --- Flatten 0: L9 ---
    // Input shape: 4x4x128
    // Output shape: 2048


    // --- Dense 0: L10 ---
    // Input shape: 2048
    // Output shape: 256


    // --- Dense 1: L11 ---
    // Input shape: 256
    // Output shape: 200


    // --- Softmax 0: L12 ---
    // Input shape: 200
    // Output shape: 200

    return model;
}


void runTests() {
    std::cout << "\n--- Running Some Tests ---" << std::endl;

    // Load an image
    fs::path imgPath("./data/image_0.bin");
    dimVec dims = {64, 64, 3};
    Array3D_fp32 img = loadArray<Array3D_fp32>(imgPath, dims);

    // Compare images
    std::cout << "Comparing image 0 to itself (max error): "
              << compareArray<Array3D_fp32>(img, img, dims)
              << std::endl
              << "Comparing image 0 to itself (T/F within epsilon " << EPSILON << "): "
              << std::boolalpha
              << compareArrayWithin<Array3D_fp32>(img, img, dims, EPSILON)
              << std::endl;

    // Test again with a modified copy
    std::cout << "\nChange a value by 1.0 and compare again" << std::endl;
    Array3D_fp32 imgCopy = allocArray<Array3D_fp32>(dims);
    copyArray<Array3D_fp32>(img, imgCopy, dims);
    imgCopy[0][0][0] += 1.0;

    std::cout << "Comparing image 0 to itself (max error): "
              << compareArray<Array3D_fp32>(img, imgCopy, dims)
              << std::endl
              << "Comparing image 0 to itself (T/F within epsilon " << EPSILON << "): "
              << std::boolalpha
              << compareArrayWithin<Array3D_fp32>(img, imgCopy, dims, EPSILON)
              << std::endl;


    // Clean up after ourselves
    freeArray<Array3D_fp32>(img, dims);
    freeArray<Array3D_fp32>(imgCopy, dims);
}


int main(int argc, char **argv) {
    // Hanlde command line arguments
    Args& args = Args::getInst();
    args.parseArgs(argc, argv);

    // Run some framework tests as an example of loading data
    runTests();

    // Base input data path (determined from current directory of where you are running the command)
    fs::path basePath("./data");

    // Build the model and allocate the buffers
    Model model = buildToyModel(basePath / "model");
    model.allocLayers<fp32>();

    // Load an image
    std::cout << "\n--- Running Infrence ---" << std::endl;
    dimVec dims = {64, 64, 3};
    
    // Construct a LayerData object from a LayerParams one
    LayerData img( {sizeof(fp32), dims, basePath / "image_0.bin"} );
    img.loadData<Array3D_fp32>();

    // Run infrence on the model
    const LayerData output = model.infrence(img, Layer::InfType::NAIVE);

    // Compare the output
    std::cout << "\n--- Comparing The Output ---" << std::endl;

    // Construct a LayerData object from a LayerParams one
    LayerData expected( { sizeof(fp32), {60, 60, 32}, basePath / "image_0_data" / "layer_0_output.bin" } );
    expected.loadData<Array3D_fp32>();
    std::cout << "Comparing expected output to model output (max error / T/F within epsilon " << EPSILON << "): \n\t"
              << expected.compare<Array3D<fp32>>(output) << " / "
              << std::boolalpha << bool(expected.compareWithin<Array3D<fp32>>(output, EPSILON))
              << std::endl;

    // Clean up
    model.freeLayers<fp32>();

    return 0;
}