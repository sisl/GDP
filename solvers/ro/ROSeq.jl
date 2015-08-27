module ROSeq

# This module contains functions for running the Richetta-Odoni dynamic model sequentially

using GDP, RO

export getScheduleFromMDP, getCapacitiesFromMDP, getROPolicy, aggregateScenarios

function getScheduleFromMDP(p::GDPParams,s::GDPState)
    # This function translates the state and schedule matrix from the MDP into a form that can be used by the Richetta-Odoni model
    empty = 1 
    S = zeros(Int64,p.dInt+empty,p.dInt+empty)
    for i = 1:size(s.am,1)
        for j = 1:size(s.am,2)
            if i-j > 0
                S[j,i] = s.am[i-1,i-j]
            end
        end
    end
    S = [zeros(Int64,1,size(S,1)); S]
    for i = 1:size(s.am,1)
        S[1,i+1] += sum(s.am[i,i+1:end])
    end
    for i = 1:p.dInt-1
        S[1,i+1] += p.sm[s.t+i,s.t+i]
    end
    for i = 1:p.dInt-1
        S[1,i+1] += sum(p.sm[s.t+i,1:s.t+i-p.dInt-1])
    end
    S[1,2] += s.h
    return S::Array{Int64}
end

function aggregateScenarios(M::Array{Int64},probs::Array{Float64})
    Mnew = unique(M,1)
    pnew = zeros(Float64,size(Mnew,1))
    ind = 0
    for i = 1:size(M,1)
        for j = 1:size(Mnew,1)
            if M[i,:] == Mnew[j,:]
                pnew[j] += 1.
                break
            end
        end
    end
    pnew /= sum(pnew)
    return Mnew::Array{Int64},pnew::Array{Float64}
end

function getCapacitiesFromMDP(p::GDPParams,s::GDPState,rng::AbstractRNG)
    # This function returns the landing capacity scenarios and transition probabilities conditioned on the current state in a form that can be used by the Richetta-Odoni model
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

function getROPolicy(p::GDPParams,rng::AbstractRNG)
    function getROAction(null,s::GDPState)
        S = getScheduleFromMDP(p,s)
        M,probs = getCapacitiesFromMDP(p,s,rng)
        M,probs = aggregateScenarios(M,probs)
        b = ROParams([0:p.dInt-1],S,M,probs,p.ca)
        return simulate(b)::GDPAction
    end
    return getROAction::Function
end

end # module
