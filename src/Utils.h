#pragma once

#include <filesystem>
#include <string>
#include <fstream>
#include <vector>
#include <type_traits>
#include <cassert>
#include <iostream>
#include <argp.h>

namespace ML {

    // --- Data Helper Functions ---
    // Floating Point Compare Epsilon
    constexpr float EPSILON = 0.0001;


    // --- Argument Parsing ---
    // GCC argument input struct
    class arguments {
        public:
            bool debug;
            bool verify;
            bool singleLayer;
            int layerNum;        
            std::filesystem::path basePath;
            char *args[1];
    };

    // Args handling class (singlton)
    class Args {
        public:
            Args(Args &other) = delete;
            void operator=(const Args &) = delete;

            static Args& getInst() {
                static Args inst;
                return inst;
            }

            void parseArgs(int argc, char **argv);

        private:
            Args() : debug(false), verify(false), singleLayer(false), layerNum(-1), basePath("") {}

        public:
            // Options
            bool debug;
            bool verify;
            bool singleLayer;
            int layerNum;        
            std::filesystem::path basePath;
            std::string version;

        private:
            struct arguments _args;
    };


// --- Implmentation ---

    // --- Data Helper Functions ---
    // Type cast helpers
    // template<typename T>
    // inline T castData(const void* data) { return (T) data; }

    // Data array allocation helpers
    // Take a raw array of dims
    template<typename T>
    void arrayAlloc(T data, const std::size_t* dims, const std::size_t dimsLen, const std::size_t dimIndex = 0) {
        // static_assert(std::is_pointer<T>(), "Cannot allocate non-pointer values (arrays)");
        typedef typename std::remove_pointer<typename std::remove_pointer<T>::type>::type* T_CLEAN;
        void** dataV = (void*) data;
        dataV = malloc(sizeof(void**) * dims[dimIndex]);
        // data = new T_CLEAN[dims[dimIndex]];
        // data = new typename std::remove_pointer<T>::type[dims[dimIndex]];
        // data = (T) new typename std::remove_all_extents<T>::type[dims[dimIndex]];
        // data = new void*[dims[dimIndex]];



        for (std::size_t i = 0; i < dims[dimIndex]; i++) {
            if (dimIndex < (dimsLen - 2)) {
                // arrayAlloc<T_CLEAN>((T_CLEAN) data[i], dims, dimsLen, dimIndex + 1);
                arrayAlloc<T_CLEAN>((T_CLEAN) dataV[i], dims, dimsLen, dimIndex + 1);
                // arrayAlloc<typename std::remove_pointer<T>::type>(data[i], dims, dimsLen, dimIndex + 1);
                // arrayAlloc<typename std::remove_extent <T>::type>(data[i], dims, dimsLen, dimIndex + 1);
                //arrayAlloc<T>(data[i], dims, dimsLen, dimIndex + 1);
            } 
            else {
                // typedef typename std::decay<T>::type T_DECAY;
                // static_cast<T_DECAY>(data)[i] = new T_DECAY[dims[dimIndex]];
                // data[i] = 0;
                dataV[i] = 0;
            }
        }
    }

    // Take a vector of dims
    template<typename T>
    inline void arrayAlloc(T data, const std::vector<std::size_t>& dims, const std::size_t dimIndex = 0) {
        static_assert(std::is_pointer<T>(), "Cannot allocate non-pointer values (arrays)");
        arrayAlloc<T>(data, dims.data(), dims.size(), dimIndex);
    }

    // Data array deallocation helpers
    // Take a raw array of dims
    template<typename T>
    void arrayFree(T data, const std::size_t* dims, const std::size_t dimsLen, const std::size_t dimIndex = 0) {
        // static_assert(std::is_pointer<T>(), "Cannot deallocate non-pointer values (arrays)");
        typedef typename std::remove_pointer<typename std::remove_pointer<T>::type>::type* T_CLEAN;
        void** dataV = (void*) data;
        
        for (std::size_t i = 0; i < dims[dimIndex]; i++) {
            if (dimIndex < dimsLen - 2) {
                arrayFree<T_CLEAN>((T_CLEAN) dataV[i], dims, dimsLen, dimIndex + 1);
            } else { 
                delete [] dataV[i];
            }
        }
    }

    // Take a vector of dims
    template<typename T>
    inline void arrayFree(T data, const std::vector<std::size_t>& dims, const std::size_t dimIndex = 0) {
        static_assert(std::is_pointer<T>(), "Cannot deallocate non-pointer values (arrays)");
        arrayFree<T>(data, dims.data(), dims.size(), dimIndex);
    }


    // --- Compare Functions ---
    // Compares two LayerData arrays of size N and returns the maximum difference
    template<typename T>
    float compare(const T data1, const T data2, const std::size_t* dims, const std::size_t dimsLen, const std::size_t dimIndex = 0) {
        // static_assert(std::is_pointer<T>(), "Cannot compare non-pointer values (arrays)");
        typedef typename std::remove_pointer<typename std::remove_pointer<T>::type>::type* T_CLEAN;
        void** dataV1 = (void*) data1;
        void** dataV2 = (void*) data2;
        double curr_diff = 0.0;
        double max_diff = 0.0;

        for (std::size_t i = 0; i < dims[dimIndex]; i++) {
            if (dimIndex < dimsLen - 2) {
                // curr_diff = compare<typename std::remove_pointer<T>::type>(data1[i], data2[i], dims, dimsLen, dimIndex + 1);
                curr_diff = compare<T_CLEAN>((T_CLEAN) dataV1[i], (T_CLEAN) dataV2[i], dims, dimsLen, dimIndex + 1);
            } else { 
                curr_diff = abs(data1[i] - data2[i]);
            }

            if (curr_diff > max_diff) {
                max_diff = curr_diff;
            }
        }

        return max_diff;
    }
    
    // Compares two LayerData arrays of size N and returns the maximum difference
    template<typename T>
    inline float compare(const T data1, const T data2, const std::vector<std::size_t> dims, const std::size_t dimIndex = 0) {
        return compare<T>(data1, data2, dims.data(), dims.size(), dimIndex);
    }


    // Performs a compare operation and checks if the max difference is within the provided epsilon
    template<typename T>
    inline bool compareWithin(const T data1, const T data2, const std::size_t* dims, const std::size_t dimsLen, float epsilon, const std::size_t dimIndex = 0) {
        return epsilon > compare<T>(data1, data2, dims, dimsLen, dimIndex);
    }

    // Performs a compare operation and checks if the max difference is within the provided epsilon
    template<typename T>
    inline bool compareWithin(const T data1, const T data2, const std::vector<std::size_t> dims, float epsilon, const std::size_t dimIndex = 0) {
        return epsilon > compareWithin<T>(data1, data2, dims.data(), dims.size(), dimIndex);
    }

    
    // --- File Data Loading ---
    // Recurrsive file data loading function for filling an allocated array with data from a file
    template<typename T>
    T loadArrayData(std::ifstream& file, T values, const std::size_t* dims, const std::size_t dimsLen, const std::size_t dimIndex = 0) {
        // static_assert(std::is_pointer<T>(), "Cannot load non-pointer values (arrays)");
        typedef typename std::remove_pointer<typename std::remove_pointer<T>::type>::type* T_CLEAN;
        void** valuesV = (void*) values;

        // Read the values and recurse if needed
        for (std::size_t i = 0; i < dims[dimIndex]; i++) {
            if (dimIndex < dimsLen - 2) {
                // We do not care about the data pointer returned here since we have that already stored in a array
                // typedef typename std::remove_pointer<T>::type T_NEXT;
                // loadArrayData<T_CLEAN>(file, values[i], dims, dimsLen, dimIndex + 1);
                loadArrayData<T_CLEAN>(file, (T_CLEAN) valuesV[i], dims, dimsLen, dimIndex + 1);
            // } else if (!file.read(reinterpret_cast<char*>(&values), sizeof(std::decay_t<T>) * dims[dimIndex])) { // Read our values
            } else if (!file.read(reinterpret_cast<char*>(valuesV), sizeof(std::decay_t<T>) * dims[dimIndex])) { // Read our values
                std::cerr << "Failed to read data values from file" << std::endl;
            }
        }

        return values;
    }

    // Entry point to loading data from a binary file into an array
    template<typename T>
    T loadArray(const std::filesystem::path& filepath, const std::vector<std::size_t>& dims) {
        static_assert(std::is_pointer<T>(), "Cannot load non-pointer values (arrays)");
        std::ifstream file(filepath, std::ios::binary); // Create and open our file
        
        if (!file.is_open()) {
            std::cerr << "Failed to open binary file " << filepath << std::endl; 
        } else {
            std::cout << "Reading data from binary file " << filepath << std::endl;
        }

        // Allocate our arrays
        T values;
        arrayAlloc<std::decay_t<T>>(values, dims.data(), dims.size(), 0);

        // Load the data
        return loadArrayData<T>(file, values, dims.data(), dims.size(), 0);
    }


} // namespace ML