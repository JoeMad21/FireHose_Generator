// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <cstdlib>
#include <vector>
#include <random>
#include <iomanip>
#include "utils.h"

#include <poputil/TileMapping.hpp>

#include <poplin/codelets.hpp>
#include <poplin/MatMul.hpp>

#include <iostream>

#define DIM 5

enum Progs {
  WRITE_INPUTS,
  REDUCTION_PROG,
  READ_RESULTS,
  NUM_PROGRAMS
};

std::vector<poplar::program::Program>
buildGraphAndPrograms(poplar::Graph &g, const utils::Options &options) {
  // Use the namespace here to make graph construction code less verbose:
  using namespace poplar;

  // Create some tensor variables. In this case they are just vectors:
  Tensor v1 = g.addVariable(FLOAT, {DIM,DIM}, "v1");
  Tensor v2 = g.addVariable(FLOAT, {DIM,DIM}, "v2");
  Tensor v3 = g.addVariable(FLOAT, {DIM,DIM}, "v3");

  // Variables need to be explcitly mapped to tiles. Put each variable on
  // a different tile (regardless of whether it is sensible in this case)

  poputil::mapTensorLinearly(g, v1);
  poputil::mapTensorLinearly(g, v2);
  poputil::mapTensorLinearly(g, v3);

  // Now can start constructing the programs. Construct a vector of
  // separate programs that can be called individually:
  std::vector<program::Program> progs(Progs::NUM_PROGRAMS);

  // Program that executes all the reduction compute sets:
  auto seq = program::Sequence();
  //seq.add(program::Execute(cs1));
  progs[REDUCTION_PROG] = seq;

  // Create streams that allow reading and writing of the variables:
  auto stream1 = g.addHostToDeviceFIFO("write_x", FLOAT, v1.numElements());
  auto stream2 = g.addHostToDeviceFIFO("write_y", FLOAT, v2.numElements());
  auto stream3 = g.addHostToDeviceFIFO("write_z", FLOAT, v3.numElements());
  auto stream4 = g.addDeviceToHostFIFO("read_z", FLOAT, v3.numElements());

  // Add a second compute set that will perform the same calculation using
  // Poplib's reduction API:
  //poplar::OptionFlags matmulOptions;
  poplin::addCodelets(g);
  auto v4 = poplin::matMul(g, v1, v2, seq, FLOAT);
  seq.add(program::Copy(v4,v3));

  progs[REDUCTION_PROG] = seq;

  // Add program which initialises the inputs. Poplar is able to merge these
  // copies for efficiency:
  progs[WRITE_INPUTS] =
      program::Sequence({program::Copy(stream1, v1), program::Copy(stream2, v2),
                         program::Copy(stream3, v3)});

  // Add a program to read back the result:
  progs[READ_RESULTS] = program::Copy(v3, stream4);

  return progs;
}

void executeGraphProgram(poplar::Device &device, poplar::Executable &exe,
                         const utils::Options &options) {
  poplar::Engine engine(std::move(exe));
  engine.load(device);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> distribution(0.0f, 100.0f);
  std::uniform_real_distribution<float> dist(0.0f,8.0f);


  std::vector<float> x(DIM*DIM);
  std::vector<float> y(DIM*DIM);
  std::vector<float> zInit(DIM*DIM);
  std::vector<float> zResult(DIM*DIM);

  for (int i = 0; i < DIM*DIM; i++) {

    x[i] = distribution(gen);
    y[i] = distribution(gen);
    zInit[i] = -1.0f;
    zResult[i] = -1.0f;
  }

  engine.connectStream("write_x", x.data());
  engine.connectStream("write_y", y.data());
  engine.connectStream("write_z", zInit.data());


  // Run program using PopLibs reduction:
  engine.connectStream("read_z", zResult.data());
  engine.run(WRITE_INPUTS);
  engine.run(REDUCTION_PROG);
  engine.run(READ_RESULTS);

  std::cout << "Matrix 1\n";
  for (int i = 0; i < x.size(); i++) {
    std::cout << std::fixed << x[i] << "\t";
    if ((i+1)%DIM == 0 && i != 0) {
      std::cout << "\n";
    }
  }
  
  std::cout << "\n";
 
  std::cout << "Matrix 2\n";
  for (int i = 0; i < y.size(); i++) {
    std::cout << std::fixed << y[i] << "\t";
    if ((i+1)%DIM == 0 && i != 0) {
      std::cout << "\n";
    }
  }

  std::cout << "\n";
  
  std::cout << "Output Matrix\n";
  for (int i = 0; i < zResult.size(); i++) {
    std::cout << std::fixed << zResult[i] << "\t";
    if ((i+1)%DIM == 0 && i != 0) {
      std::cout << "\n";
    }
  }

}

int main(int argc, char **argv) {
  try {
    auto options = utils::parseOptions(argc, argv);
    auto device = utils::getDeviceFromOptions(options);
    poplar::Graph graph(device.getTarget());

    // If we are loading the graph program we do not need
    // to construct it (which can be time consuming
    // for large programs):
    std::vector<poplar::program::Program> progs;
    if (!options.loadExe) {
      progs = buildGraphAndPrograms(graph, options);
    }

    auto exe = utils::compileOrLoadExe(graph, progs, options);

    if (options.saveExe && !options.loadExe) {
      auto outf = std::ofstream(getExeFileName(options));
      exe.serialize(outf);
    }

    executeGraphProgram(device, exe, options);

  } catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
