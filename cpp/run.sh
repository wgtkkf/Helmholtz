#!/bin/bash

# Compile the files
echo "### Compiling.             ###"
g++ -std=c++20 -I /usr/include/eigen3 main.cpp function.cpp -o exe

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "### Build successful.      ###"
    ./exe
else
    echo "Build failed."
fi
