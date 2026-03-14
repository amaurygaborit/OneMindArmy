#include <iostream>
#include <string>
#include <exception>
#include <cstdlib>
#include <yaml-cpp/yaml.h>
#include <xmmintrin.h>
#include <pmmintrin.h>

#include "bootstrap/GameTypeRegistry.hpp"

int main(int argc, char* argv[])
{
    std::cout << "===================================\n"
        << "===        One Mind Army        ===\n"
        << "===================================\n" << std::endl;

    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <config.yaml> [options]\n"
            << "Options:\n"
            << "  --mode <play|train>    Set the execution mode (default: play)\n"
            << "  --model <model_file.plan> [REQUIRED] Set the TensorRT model file name\n"
            << std::endl;
        return EXIT_FAILURE;
    }

    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    std::string configFile = argv[1];
    std::string runMode = "play";
    std::string modelFileName = "";

    // 1. Parsing des arguments CLI
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

    if (runMode != "export-meta" && modelFileName.empty()) {
        std::cerr << "[Error] You must specify a model file using --model (e.g., --model v0.plan)\n";
        return EXIT_FAILURE;
    }

    // 2. Safe Execution Block
    try
    {
        std::cout << "[System] Loading configuration from: " << configFile << "...\n";
        YAML::Node config = YAML::LoadFile(configFile);

        if (!config["name"]) {
            throw std::runtime_error("Configuration file is missing the required 'name' field.");
        }
        std::string gameName = config["name"].as<std::string>();

        // 3. Construction Automatique du chemin du modèle
        std::string finalModelPath = "models/" + gameName + "/" + modelFileName;

        std::cout << "[System] Initializing game module: [" << gameName << "]\n"
            << "[System] Mode: [" << runMode << "]\n"
            << "[System] Model Path: [" << finalModelPath << "]\n";

        // 4. Lancement (On passe les arguments explicitement !)
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