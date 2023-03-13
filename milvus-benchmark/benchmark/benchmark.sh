#!/bin/bash

# Check that two arguments have been provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 vector_num vector_dim"
    exit 1
fi

# Store arguments in variables
vector_num="$1"
vector_dim="$2"

# Execute a.py with provided arguments
echo "Execute collection prepare"
python3 collection_prepare.py 127.0.0.1 "$vector_num" "$vector_dim"

# # Execute b.py with vector_dim argument
echo "Execute benchmark"
python3 go_benchmark.py 127.0.0.1 "$vector_dim" ./benchmark

# release and drop collection
echo "Clear collections"
python3 clear_collection.py 127.0.0.1 random_benchmark_collection

# Create directory named "vec{vector_num}_{vector_dim}" in "./log"
mkdir ./log/vec${vector_num}_${vector_dim}

# move generated files into the log dir
echo "Move generated logs into log dir"
mv /tmp/collection_prepare.log ./log/vec${vector_num}_${vector_dim}
mv *.log ./log/vec${vector_num}_${vector_dim}