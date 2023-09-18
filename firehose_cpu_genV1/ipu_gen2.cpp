#include "ipu_gen.hpp"

enum Progs {
  WRITE_INPUTS,
  CONSUMPTION_TASK,
  READ_RESULTS,
  NUM_PROGRAMS
};

class GraphTensors {
  
    private:
        int num_tensors;

        std::vector<poplar::Tensor> tensors;
        std::vector<tuple <unsigned int, unsigned int>> dimensions;
        std::vector<poplar::type> tensor_types;
        std::vector<std::string> tensor_dbs;
  
    public:
        GraphTensors (poplar::Graph &g) {
    
            this->num_tensors = 3;

            this->tensor_dbs.push_back("Streamed Matrix (Multiplicand)");
            //Note that this stream should belong to the back-end
            this->tensor_dbs.push_back("Consumption Matrix (Multiplier)");
            this->tensor_dbs.push_back("Output Matrix (Result)");

            for (int i = 0; i < num_tensors; i++) {
                this->dimensions.push_back(std::tuple<int,int>{5,5});
                this->tensor_types.push_back(poplar::type FLOAT);
        
                this->tensors.push_back( g.addVariable( 
                    this->tensor_types[i],
                    {get<0>(this->dimensions[i]), get<1>(this->dimensions[i])},
                    specs.debug_contexts[i] ) );
                poplar::poputil::mapTensorLinearly(g, tensors[i]);
            }
        }

        void addTensor (poplar::Graph &g, std::string tensor_db, poplar::type tensor_type, int dim1, int dim2) {
            this->num_tensors++;

            this->tensor_dbs.push_back(db_con);

            this->tensors.push_back( g.addVariable( 
                    this->tensor_type,
                    {dim1, dim2},
                    tensor_db ) );
        }

        poplar::Tensor getTensor(int index) {
            return this->tensors[index];
        }
}

class GraphStreams {

  private:
    int num_streams;

    std::vector<poplar::DataStream> strms;
    std::vector<std::string> strm_srcs;
    std::vector<std::string> strm_dests;
    std::vector<std::string> strm_dbs;
    std::vector<poplar::type> strm_types;
    std::vector<int> strm_lengths;
    std::vector<bool> strm_dirs; // 0 = CPU to IPU, 1 = IPU to CPU

  public:
    GraphStreams(poplar::Graph &g) {
      this->num_streams = 3;
      
      this->strm_dbs.push_back("Source Stream")
      this->strm_srcs.push_back("CPU");
      this->strm_dests.push_back("IPU");
      this->strm_lengths.push_back(25);
      this->strm_types.push_back(poplar::type FLOAT);
      this->strm_dirs.push_back(0);
      this->strms.push_back( g.addHostToDeviceFIFO(strm_dbs[0], strm_types[0], strm_lengths[0]) );

      //Note that this stream should belong to the back-end
      this->strm_dbs.push_back("Consumption Stream")
      this->strm_srcs.push_back("CPU");
      this->strm_srcs.push_back("IPU");
      this->strm_lengths.push_back(25);
      this->strm_types.push_back(poplar::type FLOAT);
      this->strm_dirs.push_back(0);
      this->strms.push_back( g.addHostToDeviceFIFO(strm_dbs[1], strm_types[1], strm_lengths[1]) );

      this->strm_dbs.push_back("Result Stream")
      this->strm_srcs.push_back("IPU");
      this->strm_dests.push_back("CPU")
      this->strm_lengths.push_back(25);
      this->strm_types.push_back(poplar::type FLOAT);
      this->strm_dirs.push_back(1);
      this->strms.push_back( g.addDeviceToHostFIFO(strm_dbs[2], strm_types[2], strm_lengths[2]) );
    }

    void addHostToDeviceStream(poplar::Graph &g, std::string strm_db, int strm_length, poplar::type strm_type ) {
        this->num_streams++;

        this->strm_srcs.push_back("CPU");
        this->strm_dests.push_back("IPU");
        this->strm_lengths.push_back(strm_length);
        this->strm_types.push_back(strm_type);
        this->strm_dirs.push_back(0);
        this->strms.push_back( g.addDeviceToHostFIFO(strm_db, strm_type, strm_length) );
    }

    void addHostToDeviceStream(poplar::Graph &g, std::string strm_db, int strm_length, poplar::type strm_type ) {
        this->num_streams++;

        this->strm_srcs.push_back("IPU");
        this->strm_dests.push_back("CPU");
        this->strm_lengths.push_back(strm_length);
        this->strm_types.push_back(strm_type);
        this->strm_dirs.push_back(1);
        this->strms.push_back( g.addHostToDeviceFIFO(strm_db, strm_type, strm_length) );
    }

    poplar::DataStream getStream(int index) {
        return this->strms[index];
    }

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

std::vector<poplar::program::Program> buildPrograms(poplar::Graph &g, const utils::Options &options, GraphTensors &gTensors, GraphStreams &gStreams) {
  
  // Now can start constructing the programs. Construct a vector of
  // separate programs that can be called individually:
  std::vector<program::Program> progs(Progs::NUM_PROGRAMS);

  // Program that executes all the reduction compute sets:
  auto seq = program::Sequence();

  // Add a second compute set that will perform the same calculation using
  poplin::addCodelets(g);
  auto mult_out = poplin::matMul(g, gTensors.getTensor(0), gTensors.getTensor(1), seq, FLOAT);
  seq.add(program::Copy(mult_out,gTensors.getTensor(2)));

  progs[CONSUMPTION_TASK] = seq;

  // Add program which initialises the inputs. Poplar is able to merge these
  // copies for efficiency:
  progs[WRITE_INPUTS] =
      program::Sequence({program::Copy(gStreams.getStreams(0), gTensors.getTensor(0)), program::Copy(gStreams.getStreams(1), gTensors.getTensor(1))});

  // Add a program to read back the result:
  progs[READ_RESULTS] = program::Copy(gTensors.getTensor(2), gStreams.getStream(2));

  return progs;

}

void executeCPUCode(int dim) {
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

  printMatrix("Multiplicand", multiplicand, multiplicand.size())
  printMatrix("Multiplier", multiplier, multiplier.size())
  //printMatrix("Result", output_result, output_result.size())

}

void executeIPUCode() {
  poplar::Engine engine(std::move(exe));
  engine.load(device);

  engine.connectStream("write_source", in_strm.data());
  engine.connectStream("write_consumption", proc_mem.data());
  engine.connectStream("write_init_result", out_strm_init.data());
  engine.connectStream("read_result", out_strm_result.data()); 

  engine.run(WRITE_INPUTS);
  engine.run(CONSUMPTION_TASK);
  engine.run(READ_RESULTS);
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