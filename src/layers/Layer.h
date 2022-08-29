#pragma once

#include <vector>
#include <filesystem>
#include "../Utils.h"

namespace ML {

    // Layer Parameter structure
    class LayerParams {
        public:
            LayerParams(const std::size_t elementSize, const std::vector<std::size_t> dims) : LayerParams(elementSize, dims, "") {}
            LayerParams(const std::size_t elementSize, const std::vector<std::size_t> dims, const std::filesystem::path filePath)
                : elementSize(elementSize), dims(dims), filePath(filePath) {}

            bool isCompatible(const LayerParams& params) const;
    
        public:
            const std::size_t elementSize;
            const std::vector<std::size_t> dims;
            const std::filesystem::path filePath;
    };


    // Layer Params factory function
    template<typename T, std::size_t D0, std::size_t... D>
    inline constexpr LayerParams getParams() {
        return LayerParams(sizeof(T), {D0, D...});
    }



    // Output data container of a layer infrence
    class LayerData {
        public:
            LayerData(const LayerParams& params) : params(params), alloced(false), data(nullptr) {}

            // is the data initialized/valid
            bool isValid() const { return data != nullptr; }
            bool isAlloced() const { return alloced; }
            const LayerParams& getParams() const { return params; }

            // Get the data pointer and cast it
            template<typename T>
            T getData() const { return (T) data; }

            // Allocate data values
            template<typename T>
            inline void allocData();

            // Clean up data values
            template<typename T>
            inline void freeData();

            // Get the max difference between two Layer Data arrays
            template<typename T>
            float compare(const LayerData& other, const std::size_t dimIndex = 0);

        private:
            LayerParams params;
            bool alloced;
            void* data;
    };



    // Base class all layers extend from
    class Layer {
        public:
            // Infrence Type
            enum class InfType {
                NAIVE,
                THREADED,
                TILED,
                SIMD
            };

            // Layer Type
            enum class LayerType {
                NONE,
                CONVOLUTIONAL,
                DENSE,
                SOFTMAX,
                MAX_POLLING
            };
        
        public:
            // Contructors        
            Layer(const LayerParams &inParams, const LayerParams &outParams, LayerType lType)
                : inParams(inParams), outParams(outParams), outData(outParams), lType(lType) {}

            // Getter Functions
            const LayerParams& getInputParams() const { return inParams; }
            const LayerParams& getOutputParams() const { return outParams; }
            const LayerData& getOutputData() const { return outData; }
            LayerType getLType() const { return lType; }
            bool isOutputBufferAlloced() const { return outData.isAlloced(); }
            bool checkDataInputCompatibility(const LayerData &data) const;

            // Abstract/Virtual Functions
            virtual void computeNaive(const LayerData &dataIn) const = 0;
            virtual void computeThreaded(const LayerData &dataIn) const = 0;
            virtual void computeTiled(const LayerData &dataIn) const = 0;
            virtual void computeSIMD(const LayerData &dataIn) const = 0;

        protected:
            template<typename T>
            void allocateOutputBuffer();

            template<typename T>
            void freeOutputBuffer();

        private:
            LayerParams inParams;

            LayerParams outParams;
            LayerData outData;

            LayerType lType;
    };


    // Allocate data values
    template<typename T>
    void LayerData::allocData() {
        if (data == nullptr && !alloced) {
            arrayAlloc<T>(data, params.dims);
            alloced = true;
        } else {
            assert(false && "Cannot allocate a data pointer that has not been allocated (LayerData)");
        }
    }

    // Clean up data values
    template<typename T>
    void LayerData::freeData() {
        if (data != nullptr && alloced) {
            arrayFree<T>(data, params.dims);
            data = nullptr;
            alloced = false;
        } else {
            assert(false && "Cannot deallocate a data pointer that has not been allocated (LayerData)");
        }
    }

    // Get the max difference between two Layer Data arrays
    template<typename T>
    float LayerData::compare(const LayerData& other, const std::size_t dimIndex) {
        LayerParams aParams = getParams();
        LayerParams bParams = other.getParams();

        // Warn if we are not comparing the same data type
        if (aParams.elementSize == bParams.elementSize) {
            std::cerr << "Comparison between two LayerData arrays with different element size (and possibly data types) is not advised ("
                        << aParams.elementSize << " and " << bParams.elementSize << ")\n";
        }
        assert(aParams.dims.size() == bParams.dims.size() && "LayerData arrays must have the same number of dimentions");
        
        // Ensure each dimention size matches
        for (std::size_t size : aParams.dims) {
            assert(size == bParams.dims.size() && "LayerData arrays must have the same size dimentions");
        }

        return compare<T>(getData<T>(), other.getData<T>(), aParams.dims.data(), aParams.dims.size());
    }

    // Allocate the layer output buffer
    template<typename T>
    void Layer::allocateOutputBuffer() {
        outData.allocData<T>();
    }

    // Deallocate the layer output buffer
    template<typename T>
    void Layer::freeOutputBuffer() {
        outData.freeData<T>();
    }
}