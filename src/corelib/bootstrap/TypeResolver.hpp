#pragma once
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <yaml-cpp/yaml.h>

struct TypeResolverBase
{
    virtual ~TypeResolverBase() = default;
    virtual void run(const YAML::Node& config) const = 0;
};

class TypeResolverRegistry
{
private:
    std::unordered_map<std::string, const TypeResolverBase*> m_resolvers;

public:
    static TypeResolverRegistry& instance()
    {
        static TypeResolverRegistry inst;
        return inst;
    }

    void registerResolver(const std::string& gameName, const TypeResolverBase* resolver)
    {
        m_resolvers[gameName] = resolver;
    }

    const TypeResolverBase& get(const std::string& gameName) const
    {
        auto it = m_resolvers.find(gameName);
        if (it == m_resolvers.end())
            throw std::runtime_error("No resolver registered for game: " + gameName);
        return *it->second;
    }

    void run(const std::string& gameName, const YAML::Node& config) const
    {
        get(gameName).run(config);
    }
};