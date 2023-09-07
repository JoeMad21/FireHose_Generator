#pragma once

//#include <cstdlib>
#include <vector>
#include <random>
//#include <iomanip>
#include "utils.h"
//#include <iostream>

#include <poputil/TileMapping.hpp>

#include <poplin/codelets.hpp>
#include <poplin/MatMul.hpp>


std::vector<poplar::program::Program> buildGraphAndPrograms(poplar::Graph &g, const utils::Options &options, long unsigned int dim);

void executeGraphProgram(poplar::Device &device, poplar::Executable &exe, const utils::Options &options, long unsigned int dim);

void launchOnIPU(long unsigned matrix_dim, int argc, char **argv);