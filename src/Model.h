#pragma once
#include <vector>
#include "layers/Layer.h"
#include "layers/Convolutional.h"
#include "layers/Dense.h"
#include "layers/MaxPooling.h"
#include "layers/Softmax.h"

namespace ML {
    class Model {
        public:
            // Constructors
            Model() : layers() {} //, checkFinal(true), checkEachLayer(false) {}

            // Functions
            const LayerData& infrence(const LayerData& inData, const Layer::InfType infType = Layer::InfType::NAIVE) const;
            const LayerData& infrenceLayer(const LayerData& inData, const int layerNum, const Layer::InfType infType = Layer::InfType::NAIVE) const;
            
            // Internal memory management
            // Allocate the internal output buffers for each layer in the model
            template<typename T>
            void allocLayers();

            // Free all layers
            template<typename T>
            void clearLayers();

            // Getter Functions
            const std::size_t getNumLayers() const { return layers.size(); }
            // const bool getCheckFinal() const { return checkFinal; }
            // const bool getCheckEachLayer() const { return checkEachLayer; }

            void addLayer(Layer* l) { layers.push_back(l); }
            // void setCheckFinal(bool val) { checkFinal = val; }
            // void setCheckEachLayer(bool val) { checkEachLayer = val; }

        private:
            std::vector<Layer*> layers;
            // bool checkFinal;
            // bool checkEachLayer;
    };

    // Allocate the internal output buffers for each layer in the model
    template<typename T>
    void Model::allocLayers() {
        for (std::size_t i = 0; i < layers.size(); i++) {

            // Virtual templated functions are not allowed, so we have this
            switch (layers[i]->getLType()) {
                case Layer::LayerType::CONVOLUTIONAL:
                    ((ConvolutionalLayer*) layers[i])->allocateLayer<T>();
                    break;
                // case Layer::LayerType::DENSE:
                //     ((DenseLayer*) layers[i])->allocateLayer<T>();
                //     break;
                // case Layer::LayerType::SOFTMAX:
                //     ((SoftmaxLayer*) layers[i])->allocateLayer<T>();
                //     break;
                // case Layer::LayerType::MAX_POOLING:
                //     ((MaxPoolingLayer*) layers[i])->allocateLayer<T>();
                //     break;
            }
        }
    }

    // Free all layers in the model
    template<typename T>
    void Model::clearLayers() {
        // Free all of the layer buffers first
        // Free the internal output buffers for each layer in the model
        for (std::size_t i = 0; i < layers.size(); i++) {
            
            // Virtual templated functions are not allowed, so we have this
            switch (layers[i]->getLType()) {
                case Layer::LayerType::CONVOLUTIONAL:
                    ((ConvolutionalLayer*) layers[i])->freeLayer<T>();
                    break;
                // case Layer::LayerType::DENSE:
                //     ((DenseLayer*) layers[i])->freeLayer<T>();
                //     break;
                // case Layer::LayerType::SOFTMAX:
                //     ((SoftmaxLayer*) layers[i])->freeLayer<T>();
                //     break;
                // case Layer::LayerType::MAX_POOLING:
                //     ((MaxPoolingLayer*) layers[i])->freeLayer<T>();
                //     break;
            }
        }

        // Free layer pointers
        for (std::size_t i = 0; i < layers.size(); i++) {
            delete layers[i];
        }

        // Remove the dangeling pointers from the array
        layers.clear();
    }
}
