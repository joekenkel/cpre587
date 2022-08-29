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
Model buildToyModel(const fs::path modelDataPath) {
    Model model;

    // --- Conv 0: L0 ---
    // Input shape: 64x64x3
    // Output shape: 60x60x32
    LayerParams conv1_inDataParam = getParams<fp32, 64, 64, 3>();
    LayerParams conv1_outDataParam = getParams<fp32, 60, 60, 32>();
    LayerParams conv1_weightParam = getParams<fp32, 5, 5, 3, 32>();
    LayerParams conv1_biasParam = getParams<fp32, 32>();
    ConvolutionalLayer conv1(conv1_inDataParam, conv1_outDataParam, conv1_weightParam, conv1_biasParam);
    model.addLayer(&conv1);

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

    model.allocLayers<fp32>();
    return model;
}


void runTests() {
    // Load an image
    // fs::path imgPath("../data/image_0.bin");
    fs::path imgPath("/mnt/c/Users/mluck/git/cpre-487-587-solutions/Labs/2/data/image_0.bin");
    Array3D<fp32> img = loadArray<Array3D<fp32>>(imgPath, {64, 64, 3});

    std::cout << "Comparing image 0 to itself: "
              << compare<Array3D<fp32>>(img, img, {64, 64, 3}) << std::endl;

    std::cout << "Comparing image 0 to itself (epsilon): "
              << compareWithin<Array3D<fp32>>(img, img, {64, 64, 3}, EPSILON) << std::endl;

}


int main(int argc, char **argv) {
    // Hanlde command line arguments
    Args& args = Args::getInst();
    args.parseArgs(argc, argv);

    // Run some framework tests
    runTests();

    // Base input data path
    fs::path basePath("../data");
    fs::path testImg1Path = basePath / "image_0.bin";
    fs::path testImg1OutputBase = basePath / "test_input_0";

    // Build the model and allocate the buffers
    Model model = buildToyModel("../data/model");
    model.allocLayers<fp32>();

    // Clean up
    model.clearLayers<fp32>();

    return 0;
}