#include "ipu_gen.hpp"

enum Progs {
  WRITE_INPUTS,
  CONSUMPTION_TASK,
  READ_RESULTS,
  NUM_PROGRAMS
};

/*
struct graphSpecs {
  int num_tensors;
  int num_ipus;
  int num_streams;

  std::vector<int> assigned_ipu;

  std::vector<poplar::Tensor> tensors;
  std::vector<tuple <unsigned int, unsigned int>> dimensions;
  std::vector<poplar::type> tensor_types;
  std::vector<std::string> tensor_dbs;

  std::vector<poplar::DataStream> streams;
  std::vector<std::string> strm_srcs;
  std::vector<std::string> strm_dests;
  std::vector<std::string> stream_dbs;
  std::vector<poplar::type> stream_types;
  std::vector<int> stream_lengths;
  std::vector<bool> stream_dirs; // 0 = CPU to IPU, 1 = IPU to CPU
  //std::vector<poplar::DataStreamType> stream_types;
  
  //std::vector<> mapping;
}

void printMatrix(std::string matrix_name, std::vector<float> matrix, int matrix_dim) {
  std::cout << matrix_name << std::endl;

  for (int i = 0; i < matrix.size(); i++) {

    std::cout << std::fixed << matrix[i] << "\t";
    
    if ( (i+1)%matrix_dim == 0 && i != 0) {
      std::endl;
    }

  }
}

void buildGraph(poplar::Graph &g, const utils::Options &options, graphSpecs &specs) {

  for (int i = 0; i < specs.num_tensors; i++) {
    specs.tensors.push_back( g.addVariable( specs.type[i], {get<0>(specs.dimensions[i]), get<1>(specs.dimensions[i])}, specs.debug_contexts[i]) )
    poplar::poputil::mapTensorLinearly(g, specs.tensors[i]);
  }

}

void buildStreams(poplar::Graph &g, const utils::Options &options, graphSpecs &specs) {
  for (int i = 0; i < specs.num_streams; i++) {
    if(specs.stream_dirs[i]) {
      g.addHostToDeviceFIFO(specs.stream_dbs[i], specs.stream_types[i], specs.stream_lengths[i])
    }
    else {
      g.addDeviceToHostFIFO(specs.stream_dbs[i], specs.stream_types[i], specs.stream_lengths[i])
    }
  }
}

std::vector<poplar::program::Program> buildPrograms(poplar::Graph &g, const utils::Options &options, graphSpecs &specs) {
  
  // Now can start constructing the programs. Construct a vector of
  // separate programs that can be called individually:
  std::vector<program::Program> progs(Progs::NUM_PROGRAMS);

  // Program that executes all the reduction compute sets:
  auto seq = program::Sequence();

  // Add a second compute set that will perform the same calculation using
  poplin::addCodelets(g);
  auto mult_out = poplin::matMul(g, specs.tensor[0], specs.tensor[1], seq, FLOAT);
  seq.add(program::Copy(mult_out,specs.tensor[2]));

  progs[CONSUMPTION_TASK] = seq;

  // Add program which initialises the inputs. Poplar is able to merge these
  // copies for efficiency:
  progs[WRITE_INPUTS] =
      program::Sequence({program::Copy(specs.streams[0], specs.tensors[0]), program::Copy(specs.streams[1], specs.tensors[1])});

  // Add a program to read back the result:
  progs[READ_RESULTS] = program::Copy(specs.tensors[2], specs.streams[2]);

  return progs;

}

void executeCPUCode(int matrix_dimensions) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> distribution(0.0f, 100.0f);

  std::vector<float> multiplicand(dim*dim);
  std::vector<float> multiplier(dim*dim);
  std::vector<float> output_init(dim*dim);
  std::vector<float> output_result(dim*dim);

  for (int i = 0; i < dim*dim; i++) {
    multiplicand[i] = distribution(gen);
    multiplier[i] = distribution(gen);
    output_init[i] = -1.0f;
    output_result[i] = -1.0f;
  }
}

void executeIPUCode() {
  poplar::Engine engine(std::move(exe));
  engine.load(device);

  engine.connectStream("write_x", in_strm.data());
  engine.connectStream("write_y", proc_mem.data());
  engine.connectStream("write_z", out_strm_init.data());
  engine.connectStream("read_z", out_strm_result.data());

  engine.run(WRITE_INPUTS);
  engine.run(CONSUMPTION_TASK);
  engine.run(READ_RESULTS);
}
*/

std::vector<poplar::program::Program> buildGraphAndPrograms(poplar::Graph &g, const utils::Options &options, long unsigned int dim) {
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

  // Create streams that allow reading and writing of the variables:
  auto stream1 = g.addHostToDeviceFIFO("write_x", FLOAT, in_strm.numElements());
  auto stream2 = g.addHostToDeviceFIFO("write_y", FLOAT, proc_mem.numElements());
  auto stream3 = g.addHostToDeviceFIFO("write_z", FLOAT, out_strm.numElements());
  auto stream4 = g.addDeviceToHostFIFO("read_z", FLOAT, out_strm.numElements());

  // Add a second compute set that will perform the same calculation using
  poplin::addCodelets(g);
  auto mult_out = poplin::matMul(g, in_strm, proc_mem, seq, FLOAT);
  seq.add(program::Copy(mult_out,out_strm));

  progs[CONSUMPTION_TASK] = seq;

  // Add program which initialises the inputs. Poplar is able to merge these
  // copies for efficiency:
  progs[WRITE_INPUTS] =
      program::Sequence({program::Copy(stream1, in_strm), program::Copy(stream2, proc_mem)});

  // Add a program to read back the result:
  progs[READ_RESULTS] = program::Copy(out_strm, stream4);

  return progs;
}

std::vector<poplar::program::Program> buildGraphAndProgramsIPU_IPU(poplar::Graph &g, const utils::Options &options, long unsigned int dim){
  // Use the namespace here to make graph construction code less verbose:
  using namespace poplar;

  // Create some tensor variables. In this case they are just vectors:
  Tensor seeds0 = g.addVariable(UNSIGNED_INT, {2}, "seeds0");
  Tensor seeds1 = g.addVariable(UNSIGNED_INT, {2}, "seeds1");
  Tensor matrix0 = g.addVariable(FLOAT, {dim,dim}, "matrix0");
  Tensor matrix1 = g.addVariable(FLOAT, {dim,dim}, "matrix1");
  Tensor out_strm = g.addVariable(FLOAT, {dim,dim}, "out_strm");

  // Variables need to be explcitly mapped to tiles. Put each variable on
  // a different tile (regardless of whether it is sensible in this case)

  poputil::mapTensorLinearly(g, seeds0);
  poputil::mapTensorLinearly(g, seeds1);
  poputil::mapTensorLinearly(g, matrix0);
  poputil::mapTensorLinearly(g, matrix1);
  poputil::mapTensorLinearly(g, out_strm);
  //return progs;

  // Now can start constructing the programs. Construct a vector of
  // separate programs that can be called individually:
  std::vector<program::Program> progs(Progs::NUM_PROGRAMS);

  // Program that executes all the reduction compute sets:
  auto seq = program::Sequence();
  //seq.add(program::Execute(cs1));

  // Create streams that allow reading and writing of the variables:
  auto stream1 = g.addHostToDeviceFIFO("seeds0", UNSIGNED_INT, seeds0.numElements());
  auto stream2 = g.addHostToDeviceFIFO("seeds1", UNSIGNED_INT, seeds1.numElements());
  auto stream3 = g.addHostToDeviceFIFO("write_out_strm", FLOAT, out_strm.numElements());
  auto stream4 = g.addDeviceToHostFIFO("read_matrix0", FLOAT, matrix0.numElements());
  auto stream5 = g.addDeviceToHostFIFO("read_matrix1", FLOAT, matrix1.numElements());
  auto stream6 = g.addDeviceToHostFIFO("read_out_strm", FLOAT, out_strm.numElements());

  // Calling Poprand and Poputils libraries
  poplin::addCodelets(g);
  poprand::addCodelets(g);

  //Generates random tensor values
  //Call function that changes shape of distribution depending on user input, for now we call uniform (future)
  matrix0 = poprand::uniform(g, &seeds0, 0, matrix0, FLOAT, 0, 100, seq);
  matrix1 = poprand::uniform(g, &seeds1, 0, matrix1, FLOAT, 0, 100, seq);

  // Multiply Matrices
  auto mult_out = poplin::matMul(g, matrix0, matrix1, seq, FLOAT);
  seq.add(program::Copy(mult_out,out_strm));

  progs[CONSUMPTION_TASK] = seq;

  // Add program which initialises the inputs. Poplar is able to merge these
  // copies for efficiency:
  progs[WRITE_INPUTS] =
      program::Sequence({program::Copy(stream1, seeds0), program::Copy(stream2, seeds1)});

  // Add a program to read back the result:
  progs[READ_RESULTS] = program::Sequence({program::Copy(matrix0, stream4), program::Copy(matrix1, stream5), program::Copy(out_strm, stream6)});

  return progs;
}

void executeGraphProgram(poplar::Device &device, poplar::Executable &exe, const utils::Options &options, long unsigned int dim) {
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
    out_strm_result[i] = -1.0f;
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

  std::cout << "Sending Row 1: ";
  int count = 2;
  for (int i = 0; i < in_strm.size(); i++) {
    std::cout << std::fixed << in_strm[i] << "\t";
    if ((i+1)%dim == 0 && i != 0) {
      std::cout << "\n";
      sleep(1);
      if (count*count <= in_strm.size()) {
        std::cout << "Sending Row " << count++ << ": ";
      }
    }
  }
  
  std::cout << "\n";
  std::cout << "\n";
 
  std::cout << "In-Memory Matrix\n";
  for (int i = 0; i < proc_mem.size(); i++) {
    std::cout << std::fixed << proc_mem[i] << "\t";
    if ((i+1)%dim == 0 && i != 0) {
      std::cout << "\n";
    }
  }

  std::cout << "\n";
  
  std::cout << "Output Matrix\n";
  for (int i = 0; i < out_strm_result.size(); i++) {
    std::cout << std::fixed << out_strm_result[i] << "\t";
    if ((i+1)%dim == 0 && i != 0) {
      std::cout << "\n";
    }
  }

}

void executeGraphProgramIPU_IPU(poplar::Device &device, poplar::Executable &exe, const utils::Options &options, long unsigned int dim, uint32_t * seeds) {
  poplar::Engine engine(std::move(exe));
  engine.load(device);



  std::vector<uint32_t> seeds0(2);
  std::vector<uint32_t> seeds1(2);
  std::vector<float> out_strm_init(dim*dim);
  std::vector<float> out_strm_result(dim*dim);
  std::vector<float> matrix0_result(dim*dim);
  std::vector<float> matrix1_result(dim*dim);

//Assigns dummy values to the output stream
  for (int i = 0; i < dim*dim; i++) {
    out_strm_init[i] = -1.0f;
    out_strm_result[i] = -1.0f;
  }
  //Assigns seeds to input vectors
  seeds0[0] = *seeds++;
  seeds0[1] = *seeds++;
  seeds1[0] = *seeds++;
  seeds1[1] = *seeds;

  
  engine.connectStream("seeds0", seeds0.data());
  engine.connectStream("seeds1", seeds1.data());
  engine.connectStream("write_out_strm", out_strm_init.data());
  engine.connectStream("read_matrix0", matrix0_result.data());
  engine.connectStream("read_matrix1", matrix1_result.data());
  engine.connectStream("read_out_strm", out_strm_result.data());


  // Run program using PopLibs reduction:
  engine.run(WRITE_INPUTS);
  engine.run(CONSUMPTION_TASK);
  engine.run(READ_RESULTS);

  std::cout << "IPU0 Matrix\n";
  for (int i = 0; i < matrix0_result.size(); i++) {
    std::cout << std::fixed << matrix0_result[i] << "\t";
    if ((i+1)%dim == 0 && i != 0) {
      std::cout << "\n";
    }
  }
  
  std::cout << "\n";
 
  std::cout << "IPU1 Matrix\n";
  for (int i = 0; i < matrix1_result.size(); i++) {
    std::cout << std::fixed << matrix1_result[i] << "\t";
    if ((i+1)%dim == 0 && i != 0) {
      std::cout << "\n";
    }
  }

  std::cout << "\n";
  
  std::cout << "Output Matrix\n";
  for (int i = 0; i < out_strm_result.size(); i++) {
    std::cout << std::fixed << out_strm_result[i] << "\t";
    if ((i+1)%dim == 0 && i != 0) {
      std::cout << "\n";
    }
  }

}

void launchOnIPU(long unsigned int matrix_dim, int argc, char **argv) {
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
        auto outf = std::ofstream(utils::getExeFileName(options));
        exe.serialize(outf);
        }

        executeGraphProgram(device, exe, options, matrix_dim);

    } catch (const std::exception &e) {
         std::cerr << "Exception: " << e.what() << "\n";
         //return EXIT_FAILURE;
    }
}

void launchOnIPU_IPU_IPU(long unsigned int matrix_dim, int argc, char **argv, uint32_t *seeds) {
    try {
         auto options = utils::parseOptions(argc, argv);
         auto device = utils::getDeviceFromOptions(options);
        poplar::Graph graph(device.getTarget());

        // If we are loading the graph program we do not need
        // to construct it (which can be time consuming
        // for large programs):
        std::vector<poplar::program::Program> progs;
        if (!options.loadExe) {
            progs = buildGraphAndProgramsIPU_IPU(graph, options, matrix_dim);
        }

        auto exe = utils::compileOrLoadExe(graph, progs, options);

        if (options.saveExe && !options.loadExe) {
        auto outf = std::ofstream(utils::getExeFileName(options));
        exe.serialize(outf);
        }

        executeGraphProgramIPU_IPU(device, exe, options, matrix_dim, seeds);

    } catch (const std::exception &e) {
         std::cerr << "Exception: " << e.what() << "\n";
         //return EXIT_FAILURE;
    }
}