# nn_from_scratch

This version uses a FHE (fully homomorphic encryption) neural network for inference. 

To build and run, execute the following:

`mkdir path/to/nn_from_scratch/build && cd path/to/nn_from_scratch/build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc) && ./NNFHE`

This will install openfhe to `build/external` and link the core FHE libraries. 

See `src/example.cpp` for a basic working version. 

If you have linker errors related to `openmp` (unlikely), then you probably don't have OpenMP headers on your machine:

`sudo apt install libomp-dev`