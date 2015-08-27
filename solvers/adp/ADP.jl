module ADP 

# This module implements a post-decision rollout policy. Monte Carlo simulations are taken from the curent state and statistics stored for each action. Documentation for post-decision rollout policies can be found in 'A generalized rollout policy framework for stochastic dynamic programming'

using MDP

export ADPParams, selectAction

typealias Depth Int16

type ADPParams
    d::Depth                        # rollout depth
    ec::Float64                     # exploration constant- governs trade-off between exploration and exploitation
    rng::AbstractRNG                # random number generator
    getPossibleActions::Function    # function that returns an array of possible actions in a state
    getAction::Function             # returns action for rollout policy
    getNextState::Function          # takes state and action as arguments and returns next state from generative model
    getReward::Function             # takes state and action as arguments and returns reward
end

type PDR{T<:Action}
    A::Array{T,1}                   # set of allowable actions
    n::Array{Int32,1}               # number of times the action has been tried
    q::Array{Reward,1}              # reward approximation for each action
    PDR{T<:Action}(A::Array{T,1}) = new(A,zeros(Int32,length(A)),zeros(Reward,length(A)))
end

function selectAction(p::ADPParams,s::State)
    # This function calls simulate and chooses the approximate best action from the reward approximations 
    pdr = PDR{Action}(p.getPossibleActions(s))
    n = (length(pdr.A)-1)*200
    for i = 1:n 
        simulate(p,pdr,s,p.d)
    end
    return pdr.A[indmax(pdr.q)]::Action # Choose action with highest apporoximate value
end

function simulate(p::ADPParams,pdr::PDR,s::State,d::Depth)
    # This function runs one iteration of rollout improvement
    i = indmax(pdr.q + p.ec.*real(sqrt(complex(log(sum(pdr.n)))./pdr.n)))
    a = pdr.A[i] # choose action with highest UCT score
    sp = p.getNextState(s,a,int64(1),p.rng)
    q = p.getReward(s,a) + rollout(p,sp,int16(d-1))
    pdr.n[i] = pdr.n[i] + one(Int32)
    pdr.q[i] += (q-pdr.q[i])/pdr.n[i]
end

function rollout(p::ADPParams,s::State,d::Depth)
    # Runs a rollout simulation using the default policy
    if d == 0
        return 0.0::Reward
    else 
        a = p.getAction(s,p.rng)
        sp = p.getNextState(s,a,int64(p.d+1-d),p.rng)
        return (p.getReward(s,a) + rollout(p,sp,int16(d-1)))::Reward
    end	
end

end # module
