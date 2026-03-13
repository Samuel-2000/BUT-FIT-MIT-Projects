#!/bin/bash

# Check for exactly one argument
if [ $# -ne 1 ]; then
  echo "Usage: $0 <node_string>" >&2
  exit 1
fi

node_str="$1"
np=${#node_str}  # Number of processes is the length of the node string

if [ $np -eq 0 ]; then
  echo "Node string must not be empty" >&2
  exit 2
fi

# Compile the program
mpic++ --prefix /usr/local/share/OpenMPI -o vuv vuv.cpp

# Run with the computed number of processes and pass the node string as an argument
mpirun --oversubscribe --prefix /usr/local/share/OpenMPI -np $np vuv "$node_str"

# Cleanup
rm -f vuv