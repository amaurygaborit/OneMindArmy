#pragma once
#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <stdexcept>

#include "../interfaces/IHandler.hpp"
#include "../model/ReplayBuffer.hpp" 

namespace Core
{
    // ============================================================================
    // META EXPORT HANDLER
    // Utility mode to resolve the "Chicken and Egg" bootstrapping paradox.
    // Exports the C++ constexpr geometry to JSON so Python can generate v0.onnx.
    // ============================================================================
    template<ValidGameTraits GT>
    class MetaExportHandler : public IHandler<GT>
    {
    public:
        USING_GAME_TYPES(GT);

    private:
        std::string m_gameName;

        void specificSetup(const YAML::Node& config) override
        {
            std::cout << "[MetaExportHandler] Setup initialized.\n";

            if (!config["name"]) {
                throw std::runtime_error("Config Error: Missing 'name' field in YAML.");
            }
            // A simple string extraction. LoadVal is reserved for bounded numerics.
            m_gameName = config["name"].as<std::string>();
        }

    public:
        MetaExportHandler() = default;
        virtual ~MetaExportHandler() = default;

        void execute() override
        {
            std::cout << "\n[MetaExport] Extracting constexpr dimensions for game: [" << m_gameName << "]\n";

            std::string dataFolder = "data/" + m_gameName;
            std::filesystem::create_directories(dataFolder);

            std::string metaPath = dataFolder + "/" + m_gameName + "_training_data.bin.meta.json";
            std::ofstream metaFile(metaPath);

            if (metaFile.is_open())
            {
                // ----------------------------------------------------------------
                // NOUVEAU : On exporte la taille totale de la structure en octets
                // (incluant le padding / alignement mémoire) pour le Dataset Python.
                // ----------------------------------------------------------------
                metaFile << "{\n"
                    << "  \"numPlayers\": " << Defs::kNumPlayers << ",\n"
                    << "  \"numPos\": " << Defs::kNumPos << ",\n"
                    << "  \"actionSpace\": " << Defs::kActionSpace << ",\n"
                    << "  \"nnInputSize\": " << Defs::kNNInputSize << ",\n"
                    << "  \"sizeofTrainingSample\": " << sizeof(TrainingSample<GT>) << "\n"
                    << "}\n";
                metaFile.close();

                std::cout << "[MetaExport] Success! Metadata locked and saved to: " << metaPath << "\n";
                std::cout << "             Struct size mapped to: " << sizeof(Core::TrainingSample<GT>) << " bytes.\n";
            }
            else
            {
                throw std::runtime_error("Fatal Error: Could not write metadata file to " + metaPath);
            }
        }
    };
}