# SQUIC_CPP


SQUIC CPP example
 
This is an example code for using libSQUIC in CPP. Here we are using Armadillo(http://arma.sourceforge.net) for simplicity ( it is not necessary).

Compile commaned ( e.g., for debuging):
g++ -std=c++11 -O0 -g -fsanitize=address   main.cpp - o main.exe -larmadillo -/Location/oF/libSQUIC  -lSQUIC

note you will need to set the path folder of libSQUIC for runtime, for example:

export DYLD_LIBRARY_PATH=/Location/Of/libSQUIC (for Mac)
export LD_LIBRARY_PATH=/Location/Of/libSQUIC   (for Linux)
