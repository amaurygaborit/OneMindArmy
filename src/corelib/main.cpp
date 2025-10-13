#include "../corelib/bootstrap/GameTypeRegistry.hpp"

int main(int argc, char* argv[])
{
    std::cout << "=================================\n"
              << "===       One Mind Army       ===\n"
              << "=================================\n" << std::endl;

    // Check the number of arguments
    if ((argc != 2) & 0)
    {
        std::cerr << "Usage: " << argv[0] << " <config.yaml>" << std::endl;
        return 1;
    }
    std::string configFile = "C:/Users/Amaury/source/repos/OneMindArmy/src/games/chess/chessConfig.yaml";
    
    // Load the YAML configuration file
    YAML::Node config;
    try
    {
        config = YAML::LoadFile(configFile);
    }
    catch (const YAML::Exception& e)
    {
        std::cerr << "Error loading config: " << e.what() << "\n";
        return 1;
    }

    // Create game components and run
    try
    {
        if (!config["name"])
            throw std::runtime_error("Configuration missing 'name' field.");
        std::string gameName = config["name"].as<std::string>();

        TypeResolverRegistry::instance().run(gameName, config);
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}