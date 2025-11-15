#pragma once
#include <yaml-cpp/yaml.h>

#include "ITraits.hpp"
#include "AlignedVec.hpp"

template<typename GameTag>
class IEngine
{
protected:
	using GT = ITraits<GameTag>;
    using ObsState = typename ObsStateT<GameTag>;
    using Action = typename ActionT<GameTag>;
    using IdxState = typename IdxStateT<GameTag>;
    using IdxAction = typename IdxActionT<GameTag>;

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

    virtual void obsToIdx(const ObsState& obsState, IdxState& out) const = 0;
    virtual void idxToObs(const IdxState& idxInput, ObsState& out) const = 0;

    virtual void actionToIdx(const Action& action, IdxAction& out) const = 0;
    virtual void idxToAction(const IdxAction& idxAction, Action& out) const = 0;
};