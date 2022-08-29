#pragma once

#include "Layer.h"
#include "../Utils.h"
#include "../Types.h"

namespace ML {
    class ConvolutionalLayer : public Layer {
        public:
            ConvolutionalLayer(const LayerParams &inParams, const LayerParams &outParams,
                               const LayerParams &weightParams, const LayerParams &biasParams)
                : Layer(inParams, outParams, LayerType::CONVOLUTIONAL), weightParam(weightParams), weightData(weightParams),
                  biasParam(biasParams), biasData(biasParams) {}

            // Getters
            const LayerParams& getWeightParams() const { return weightParam; }
            const LayerParams& getBiasParams() const { return biasParam; }
            const LayerData& getInputData() const { return weightData; }
            const LayerData& getOutputData() const { return biasData; }

            // Allocate all resources needed for the layer
            template<typename T>
            void allocateLayer() {
                Layer::allocateOutputBuffer<Array3D<T>>();
                weightData.allocData<Array4D<T>>();
                biasData.allocData<Array1D<T>>();
            }

            // Load all of the required data for the layer
            template<typename T>
            void loadLayer() {
                // TODO: Your Code Here...

            }

            // Fre all resources allocated for the layer
            template<typename T>
            void freeLayer() {
                Layer::freeOutputBuffer<Array3D<T>>();
                weightData.freeData<Array4D<T>>();
                biasData.freeData<Array1D<T>>();
            }

            // Virtual functions
            virtual void computeNaive(const LayerData &dataIn) const override;
            virtual void computeThreaded(const LayerData &dataIn) const override;
            virtual void computeTiled(const LayerData &dataIn) const override;
            virtual void computeSIMD(const LayerData &dataIn) const override;

        private:
            LayerParams weightParam;
            LayerData weightData;

            LayerParams biasParam;
            LayerData biasData;
    };

}