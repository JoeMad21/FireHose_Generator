#pragma once

#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>

#include <boost/program_options.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>

namespace utils {

  struct Options {
  std::size_t devices;
  std::size_t con_task;
  std::size_t source;
  std::size_t dimensions;
  std::size_t numIpus;
  std::string exeName;
  std::string profileName;
  bool useIpuModel;
  bool saveExe;
  bool loadExe;
  bool profile;
  };

  Options parseOptions(int argc, char **argv);
  std::string getExeFileName(const Options &options);

  poplar::Executable compileOrLoadExe(poplar::Graph &graph,
                 const std::vector<poplar::program::Program> &progs,
                 const Options &options);

  poplar::Device getIpuHwDevice(std::size_t numIpus);
  poplar::Device getIpuModelDevice(std::size_t numIpus);
  poplar::Device getDeviceFromOptions(const Options &options);

}