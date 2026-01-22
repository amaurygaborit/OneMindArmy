#pragma once
#include <yaml-cpp/yaml.h>

#include "ITraits.hpp"
#include "AlignedVec.hpp"
#include "../bootstrap/GameConfig.hpp"

template<typename GameTag>
class IEngine
{
protected:
    using GT = ITraits<GameTag>;
    using ObsState = typename GT::ObsState;
    using Action = typename GT::Action;
    using FactState = typename FactStateT<GameTag>;
    using FactAction = typename Fact<GameTag>;

    virtual void specificSetup(const YAML::Node& config) = 0;

public:
    virtual ~IEngine() = default;
    void setup(const YAML::Node& config)
    {
        specificSetup(config);
	};

    virtual void getInitialState(const size_t player, ObsState& out) const = 0;

    virtual size_t getCurrentPlayer(const ObsState& obsState) const = 0;

    virtual void getValidActions(const ObsState& obsState, AlignedVec<Action>& out) const = 0;
    virtual bool isValidAction(const ObsState& obsState, const Action& action) const = 0;
    virtual void applyAction(const Action& action, ObsState& out) const = 0;
    virtual bool isTerminal(const ObsState& obsState, AlignedVec<float>& out) const = 0;

    // --- Converters ---
    virtual void stateToFacts(const ObsState& obsState, FactState& out) const = 0;
    virtual void actionToFact(const Action& action, const ObsState& obsState, FactAction& out) const = 0;

    virtual void idxToAction(uint32_t idxAction, Action& out) const = 0;
    virtual uint32_t actionToIdx(const Action& action) const = 0;
};