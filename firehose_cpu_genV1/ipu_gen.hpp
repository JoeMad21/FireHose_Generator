#pragma once

//#include <cstdlib>
#include <vector>
#include <random>
//#include <iomanip>
#include "utils.h"
//#include <iostream>
#include <unistd.h>

#include <poputil/TileMapping.hpp>

#include <poplin/codelets.hpp>
#include <poplin/MatMul.hpp>

#include <poprand/codelets.hpp>
#include <poprand/RandomGen.hpp>


std::vector<poplar::program::Program> buildGraphAndPrograms(poplar::Graph &g, const utils::Options &options, long unsigned int dim);
std::vector<poplar::program::Program> buildGraphAndProgramsIPU_IPU(poplar::Graph &g, const utils::Options &options, long unsigned int dim);


void executeGraphProgram(poplar::Device &device, poplar::Executable &exe, const utils::Options &options, long unsigned int dim);
void executeGraphProgramIPU_IPU(poplar::Device &device, poplar::Executable &exe, const utils::Options &options, long unsigned int dim, uint32_t * seeds);

void launchOnIPU(long unsigned matrix_dim, int argc, char **argv);
void launchOnIPU_IPU_IPU(long unsigned int matrix_dim, int argc, char **argv, uint32_t *seeds);