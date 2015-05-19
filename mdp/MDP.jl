module MDP

export GenerativeModel, Params, State, Action, Reward, simulate 

type GenerativeModel
    getInitialState::Function 
    getNextState::Function   
    getReward::Function     
end

typealias Policy Function
typealias Reward Float64
typealias Params Any

abstract State
abstract Action

function simulate(model::GenerativeModel,p::Params,policy::Policy,nSteps::Int,rng::AbstractRNG)
    # This function simulates the model for nSteps using the specified policy and returns the total simulation reward 
    r = 0.
    s = model.getInitialState(rng)
    for i = 1:nSteps
        a = policy(p,s)
        s = model.getNextState(s,a,rng)
        r += model.getReward(s,a)
    end
    return r::Float64
end

end # module
