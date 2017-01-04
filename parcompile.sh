
export LD_LIBRARY_PATH="/usr0/home/mkshirsa/research/packages/shogun/usr/local/lib"
g++ -O2 -std=c++0x -fopenmp -I/usr0/home/mkshirsa/research/Downloads/shogun-3.0.0/src/ RunAltLS.cpp -L$LD_LIBRARY_PATH -lshogun -lconfig++ -o bsl_mtl

#export LD_LIBRARY_PATH="/usr0/home/research/packages/shogun/usr/local/lib"
#g++ -O2 -std=c++0x -fopenmp -I/usr0/home/research/Downloads/shogun-3.0.0/src/ RunAltLS.cpp -L$LD_LIBRARY_PATH -lshogun -lconfig++ -o bsl_mtl

