#pragma once
#include <yaml-cpp/yaml.h>

#include "ITraits.hpp"
#include "AlignedVec.hpp"

template<typename GameTag>
class IEngine
{
protected:
    uint16_t m_maxValidActions = 0;

    virtual void specificSetup(const YAML::Node& config) = 0;

public:
    virtual ~IEngine() = default;
    void setup(const YAML::Node& config)
    {
        if (!config["common"]["engine"]["maxValidActions"])
            throw std::runtime_error("Configuration missing 'common.engine.maxValidActions' field");

        m_maxValidActions = config["common"]["engine"]["maxValidActions"].as<uint16_t>();

        specificSetup(config);
	};

    uint16_t getMaxValidActions() const { return m_maxValidActions; };

    virtual void getInitialState(ObsStateT<GameTag>& out) = 0;
    virtual uint8_t getCurrentPlayer(const ObsStateT<GameTag>& obsState) = 0;
    virtual void getValidActions(const ObsStateT<GameTag>& obsState, AlignedVec<ActionT<GameTag>>& out) = 0;
    virtual bool isValidAction(const ObsStateT<GameTag>& obsState, const ActionT<GameTag>& action) = 0;
    virtual void applyAction(const ActionT<GameTag>& action, ObsStateT<GameTag>& out) = 0;
    virtual bool isTerminal(const ObsStateT<GameTag>& obsState, AlignedVec<float>& out) = 0;

    virtual void obsToIdx(const ObsStateT<GameTag>& obsState, IdxStateT<GameTag>& out) = 0;
    virtual void idxToObs(const IdxStateT<GameTag>& idxInput, ObsStateT<GameTag>& out) = 0;

    virtual void actionToIdx(const ActionT<GameTag>& action, IdxActionT& out) = 0;
    virtual void idxToAction(const IdxActionT& idxAction, ActionT<GameTag>& out) = 0;
};