#!/bin/bash
set -e
set -x

cd "$( dirname "${BASH_SOURCE[0]}" )"

rm -rf build
mkdir build
cd build
# for release build: -DCMAKE_BUILD_TYPE=Release
cmake .. $@
make
./vector_add_example
