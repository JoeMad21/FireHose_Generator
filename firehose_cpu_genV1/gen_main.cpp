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

enum Progs {
  WRITE_INPUTS,
  CONSUMPTION_TASK,
  READ_RESULTS,
  NUM_PROGRAMS
};

std::vector<poplar::program::Program>
buildGraphAndPrograms(poplar::Graph &g, const utils::Options &options, long unsigned int dim) {
  // Use the namespace here to make graph construction code less verbose:
  using namespace poplar;

  // Create some tensor variables. In this case they are just vectors:
  Tensor in_strm = g.addVariable(FLOAT, {dim,dim}, "in_strm");
  Tensor proc_mem = g.addVariable(FLOAT, {dim,dim}, "proc_mem");
  Tensor out_strm = g.addVariable(FLOAT, {dim,dim}, "out_strm");

  // Variables need to be explcitly mapped to tiles. Put each variable on
  // a different tile (regardless of whether it is sensible in this case)

  poputil::mapTensorLinearly(g, in_strm);
  poputil::mapTensorLinearly(g, proc_mem);
  poputil::mapTensorLinearly(g, out_strm);

  // Now can start constructing the programs. Construct a vector of
  // separate programs that can be called individually:
  std::vector<program::Program> progs(Progs::NUM_PROGRAMS);

  // Program that executes all the reduction compute sets:
  auto seq = program::Sequence();
  //seq.add(program::Execute(cs1));
  progs[CONSUMPTION_TASK] = seq;

  // Create streams that allow reading and writing of the variables:
  auto stream1 = g.addHostToDeviceFIFO("write_x", FLOAT, in_strm.numElements());
  auto stream2 = g.addHostToDeviceFIFO("write_y", FLOAT, proc_mem.numElements());
  auto stream3 = g.addHostToDeviceFIFO("write_z", FLOAT, out_strm.numElements());
  auto stream4 = g.addDeviceToHostFIFO("read_z", FLOAT, out_strm.numElements());

  // Add a second compute set that will perform the same calculation using
  poplin::addCodelets(g);
  auto mult_out = poplin::matMul(g, in_strm, proc_mem, seq, FLOAT);
  seq.add(program::Copy(mult_out,out_strm));

  // Add program which initialises the inputs. Poplar is able to merge these
  // copies for efficiency:
  progs[WRITE_INPUTS] =
      program::Sequence({program::Copy(stream1, in_strm), program::Copy(stream2, proc_mem)});

  // Add a program to read back the result:
  progs[READ_RESULTS] = program::Copy(out_strm, stream4);

  return progs;
}

void executeGraphProgram(poplar::Device &device, poplar::Executable &exe,
                         const utils::Options &options, long unsigned int dim) {
  poplar::Engine engine(std::move(exe));
  engine.load(device);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> distribution(0.0f, 100.0f);
  std::uniform_real_distribution<float> dist(0.0f,8.0f);


  std::vector<float> in_strm(dim*dim);
  std::vector<float> proc_mem(dim*dim);
  std::vector<float> out_strm_init(dim*dim);
  std::vector<float> out_strm_result(dim*dim);

  for (int i = 0; i < dim*dim; i++) {

    in_strm[i] = distribution(gen);
    proc_mem[i] = distribution(gen);
    out_strm_init[i] = -1.0f;
    out_stream_result[i] = -1.0f;
  }

  engine.connectStream("write_x", in_strm.data());
  engine.connectStream("write_y", proc_mem.data());
  engine.connectStream("write_z", out_strm_init.data());
  engine.connectStream("read_z", out_strm_result.data());


  // Run program using PopLibs reduction:
  engine.run(WRITE_INPUTS);
  engine.run(CONSUMPTION_TASK);
  engine.run(READ_RESULTS);

  std::cout << "Input Matrix\n";
  for (int i = 0; i < x.size(); i++) {
    std::cout << std::fixed << x[i] << "\t";
    if ((i+1)%dim == 0 && i != 0) {
      std::cout << "\n";
    }
  }
  
  std::cout << "\n";
 
  std::cout << "In-Memory Matrix\n";
  for (int i = 0; i < y.size(); i++) {
    std::cout << std::fixed << y[i] << "\t";
    if ((i+1)%dim == 0 && i != 0) {
      std::cout << "\n";
    }
  }

  std::cout << "\n";
  
  std::cout << "Output Matrix\n";
  for (int i = 0; i < zResult.size(); i++) {
    std::cout << std::fixed << zResult[i] << "\t";
    if ((i+1)%dim == 0 && i != 0) {
      std::cout << "\n";
    }
  }

}

int main(int argc, char **argv) {

  int device = -1; // 0 = IPU
  std::cout << "Welcome to the FireHose Generator. What device are you targeting today?" << std::endl;
  std::cout << "1. Graphcore IPU" << std::endl;
  std::cout << "2. UPMEM DPU" << std::endl;
  std::cout << "Selected Device: ";
  std::cin >> device;

  switch(device) {
    case -1:
      std::cout << "No device selected, ending program" << std::endl;
      return 0;
      break;
    case 0:
      std::cout << "No device selected, ending program" << std::endl;
      return 0;
      break;
    default:
      std::cout << "IPU selected" << std::endl;
  }

  int consumption_task = -1;
  std::cout << "What consumption task would you like to do on the back-end?" << std::endl;
  std::cout << "1. Matrix multiplication" << std::endl;
  std::cout << "2. Graph task" << std::endl;
  std::cout << "3. Hashing" << std::endl;
  std::cout << "Consumption Task: ";
  std::cin >> consumption_task;

  switch(consumption_task) {
    case -1:
      std::cout << "No consumption task selected, ending program" << std::endl;
      return 0;
      break;
    case 0:
      std::cout << "No consumption task selected, ending program" << std::endl;
      return 0;
      break;
    default:
      std::cout << "Matrix multiplication selected" << std::endl;
  }

  long unsigned int matrix_dim = 0;
  std::cout << "What dimensions would you like for your square matrix? (NxN)" << std::endl;
  std::cout << "Enter N: ";
  std::cin >> matrix_dim;

  try {
    auto options = utils::parseOptions(argc, argv);
    auto device = utils::getDeviceFromOptions(options);
    poplar::Graph graph(device.getTarget());

    // If we are loading the graph program we do not need
    // to construct it (which can be time consuming
    // for large programs):
    std::vector<poplar::program::Program> progs;
    if (!options.loadExe) {
      progs = buildGraphAndPrograms(graph, options, matrix_dim);
    }

    auto exe = utils::compileOrLoadExe(graph, progs, options);

    if (options.saveExe && !options.loadExe) {
      auto outf = std::ofstream(getExeFileName(options));
      exe.serialize(outf);
    }

    executeGraphProgram(device, exe, options, matrix_dim);

  } catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
