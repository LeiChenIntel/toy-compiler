#include "Utils/Utility.h"

#include <iostream>
#include <llvm/Support/CommandLine.h>

Platform getPlatformFromCmd(int argc, char *argv[]) {
  // Define command line option in MLIR. When run with helper -h, the description is shown as
  // --toy-platform=<device-name> - Target platform, e.g., device1, device2
  static llvm::cl::opt<std::string> platform("toy-platform",
                                             llvm::cl::desc("Target platform, e.g., device1, device2"),
                                             llvm::cl::value_desc("device-name"),
                                             llvm::cl::init("device1"));
  // llvm::cl::ParseCommandLineOptions(argc, argv); should be called after registry
  // Need to parse command line manually here
  std::string deviceName = "device1";
  for (int i = 1; i < argc; ++i) {
    // argv[0] is always the program name
    auto strArgv = std::string(argv[i]);
    if (strArgv.find("toy-platform") != std::string::npos) {
      const auto strDeviceStart = strArgv.find("toy-platform") + std::string("toy-platform").length() + 1; // +1 for '='
      deviceName = strArgv.substr(strDeviceStart, std::distance(strArgv.begin() + strDeviceStart, strArgv.end()));
      break;
    }
  }

  if (deviceName == "device1") {
    return Platform::Device1;
  }
  if (deviceName == "device2") {
    return Platform::Device2;
  }
  llvm::errs() << "Error: Unknown device: " << deviceName << "\n";
  std::exit(1);
}
