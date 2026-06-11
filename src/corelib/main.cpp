#include <iostream>
#include <string>
#include <exception>
#include <cstdlib>
#include <yaml-cpp/yaml.h>
#include <xmmintrin.h>
#include <pmmintrin.h>

#include "bootstrap/GameTypeRegistry.hpp"

// ============================================================================
// MAIN ENTRY POINT
// Initializes the hardware environment, parses Command Line Arguments (CLI), 
// and hands off execution to the polymorphic GameRegistry.
// ============================================================================
int main(int argc, char* argv[])
{
    std::cout << "===================================\n"
        << "===        One Mind Army        ===\n"
        << "===================================\n" << std::endl;

    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <config.yaml> [options]\n"
            << "Options:\n"
            << "  --mode <play|train|export-meta>  Set the execution mode (default: play)\n"
            << "  --model <model_file.plan> [REQUIRED] Set the TensorRT model file name\n"
            << std::endl;
        return EXIT_FAILURE;
    }

    // CPU HARDWARE OPTIMIZATION: Flush-to-Zero (FTZ) & Denormals-Are-Zero (DAZ)
    // Prevents severe CPU pipeline stalls caused by extreme precision floating-point 
    // underflows (subnormal numbers) during Neural Network Softmax calculations.
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    std::string configFile = argv[1];
    std::string runMode = "play";
    std::string modelFileName = "";

    // 1. CLI Argument Parsing
    // Extracts runtime execution context independently of the YAML configuration.
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) {
            runMode = argv[++i];
        }
        else if (arg == "--model" && i + 1 < argc) {
            modelFileName = argv[++i];
        }
        else {
            std::cerr << "[Warning] Unknown or incomplete argument ignored: " << arg << "\n";
        }
    }

    // The 'export-meta' pipeline extracts pure C++ geometry without requiring an active model.
    if (runMode != "export-meta" && modelFileName.empty()) {
        std::cerr << "[Error] You must specify a model file using --model (e.g., --model v0.plan)\n";
        return EXIT_FAILURE;
    }

    // 2. Safe Execution Block
    // Traps and logs all setup-phase exceptions (e.g., missing YAML nodes) before 
    // handing over control to the multithreaded handlers.
    try
    {
        std::cout << "[System] Loading configuration from: " << configFile << "...\n";
        YAML::Node config = YAML::LoadFile(configFile);

        // The game name acts as the universal lookup key for the GameRegistry
        if (!config["name"]) {
            throw std::runtime_error("Configuration file is missing the required 'name' field.");
        }
        std::string gameName = config["name"].as<std::string>();

        // 3. Automated Path Resolution
        // Binds the relative model name to the specific game's directory.
        std::string finalModelPath = "models/" + gameName + "/" + modelFileName;

        std::cout << "[System] Initializing game module: [" << gameName << "]\n"
            << "[System] Mode: [" << runMode << "]\n"
            << "[System] Model Path: [" << finalModelPath << "]\n";

        // 4. Execution Dispatch
        // Looks up the compiled GameBootstrapper associated with the parsed game name
        // and triggers dependency injection.
        Core::GameRegistry::instance().run(gameName, config, runMode, finalModelPath);
    }
    catch (const std::exception& e)
    {
        std::cerr << "[Critical Error] " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "\n[System] Engine shut down successfully.\n";
    return EXIT_SUCCESS;
}