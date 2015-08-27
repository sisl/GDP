module BallSeq

# This module contains functions for running the Ball model sequentially

using GDP, Ball

export getScheduleFromMDP, getCapacitiesFromMDP, getBallPolicy

function getScheduleFromMDP(p::GDPParams,s::GDPState)
    # This function translates the state and schedule matrix from the MDP into a form that can be used by the Ball model
    empty = 1
    S = zeros(Int64,p.dInt,2) # first column not exempt, second column exempt
    for i = 1:size(S,1)
        S[i,1] = sum(s.am[i,1:i]) + sum(p.sm[s.t+i,1:s.t+1-p.dInt])
        S[i,2] = sum(s.am[i,i+1:end]) + p.sm[s.t+i,s.t+i]
    end
    S[1,2] += s.h
    S = [S, zeros(Int64,empty,2)]
    return S::Array{Int64}
end

function getCapacitiesFromMDP(p::GDPParams,s::GDPState,rng::AbstractRNG)
    # This function returns the landing capacity scenarios and transition probabilities conditioned on the current state in a form that can be used by the Ball model
    n = 100 # number to sample
    empty = 1 
    M = zeros(Int64,n,p.dInt+empty) # capacity scenarios
    probs = ones(Float64,n)
    probs /= sum(probs)
    for i = 1:n
        for j = 1:size(M,2)
            if j == 1
                M[i,j] = p.getAAR(int64(s.aar),s.t+1,j,rng)
            elseif j == size(M,2)
                M[i,j] = 1000000 # basically infinity
            else
                M[i,j] = p.getAAR(int64(M[i,j-1]),s.t+1,j,rng)
            end
        end
    end
    return M::Array{Int64}, probs::Array{Float64}
end

function getBallPolicy(p::GDPParams,rng::AbstractRNG)
    function getBallAction(null,s::GDPState)
        S = getScheduleFromMDP(p,s)
        M,probs = getCapacitiesFromMDP(p,s,rng)
        b = BallParams(S,M,probs,p.ca)
        return simulate(b)::GDPAction
    end
    return getBallAction::Function
end

end # module
