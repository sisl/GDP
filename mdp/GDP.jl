module GDP

# This module is an MDP model for the Single Airport Ground Holding Problem 

using MDP, DataFrames, Iterators

export GDPParams, GDPState, GDPAction, getGenerativeModel, getGenerativeModelADP, getPolicyFunctions

typealias Flights Int16
typealias Interval Int16
typealias Reward Float64

type GDPParams
    dInt::Interval                        # number of delay intervals
    dOpt::Symbol                          # delay option, for getDelayMatrix
    sm::Array{Flights,2}                  # schedule matrix
    aars::Array{Flights,2}                # vector of possible AARs
    firstAAR::Array{Float64,2}            # vector of probabilities for first AAR
    ca::Reward                            # cost of one unit of air delay
    getAAR::Function                      # generative model that returns the next AAR 
end

immutable GDPState <: State
    t::Interval                           # time interval
    aar::Flights                          # aircraft arrival rate
    h::Flights                            # number holding
    am::Array{Flights,2}                  # arrival demand matrix
end

type GDPAction <: Action
    td::Array{Flights,2}                  # number to delay in each interval
end

function getGenerativeModel(p::GDPParams)
    return GenerativeModel(getInitialStateFunction(p),getNextStateFunction(p),getRewardFunction(p))::GenerativeModel
end

function getGenerativeModelADP(p::GDPParams)
    return GenerativeModel(getInitialStateFunction(p),getNextStateFunctionADP(p),getRewardFunction(p))::GenerativeModel
end

function getPolicyFunctions(p::GDPParams)
    return getActionFunction(p)::Function, getPossibleActionsFunction(p)::Function
end

function getInitialStateFunction(p::GDPParams)
    # This function returns the initial state function
    function getInitialState(rng::AbstractRNG)
        sh = zero(Int16) # start hour
        iam = zeros(Flights,p.dInt,p.dInt)
        for i = 1:p.dInt
            for j = 1:p.dInt
                if sh+i-j > 0 
                    iam[i,j] = p.sm[sh+i,sh+i-j] # get initial arrival demand matrix 
                else
                    iam[i,j] = zero(Int16)
                end
            end
        end
        aar = getFirstAAR(p,rng)
        return GDPState(sh,aar,zero(Int16),iam)::State
    end
    return getInitialState::Function
end

function getFirstAAR(p::GDPParams,rng::AbstractRNG)
    # This is a generative model for drawing a first AAR
    draw = rand(rng)
    temp = 0.
    for i = 1:length(p.firstAAR)
        temp += p.firstAAR[i]
        if draw < temp
            return p.aars[i]::Flights
        end
    end
end

function getRewardFunction(p::GDPParams)
    # This function returns the reward function
    cg = 1. # cost one unit of ground delay
    function getReward(s::State,a::Action)
        return -(cg*sum(a.td)+p.ca*s.h)::Reward
    end
    return getReward::Function
end

function getNextStateFunctionADP(p::GDPParams)
    # This function returns the next state function
    function getNextState(s::State,a::Action,d::Int,rng::AbstractRNG)
        dm = getDelayMatrix(p,s,a) #get delay matrix
        nam = zeros(Flights,p.dInt,p.dInt)
        for i = 1:p.dInt  # put together new arrival demand matrix
            for j = 1:p.dInt
                if i+1 < j && i != p.dInt
                    nam[i,j] = s.am[i+1,j] 
                elseif i+1 == j && i != p.dInt
                    nam[i,j] = s.am[i+1,j] - dm[i+1,j] 
                elseif i+1 > j && i != p.dInt
                    nam[i,j] = s.am[i+1,j] + dm[i,j] - dm[i+1,j] 
                else
                    nam[i,j] = p.sm[s.t+1+p.dInt,s.t+1+p.dInt-j] + dm[p.dInt,j] 
                end
            end
        end
        t = int16(s.t+1) # update time interval
        aar = p.getAAR(int64(s.aar),s.t+1,d,rng) # update aar
        h = int16(max(sum(s.am[1,:])+p.sm[t,t]+sum(p.sm[t,1:t-p.dInt-1])-dm[1,1]+s.h-aar,0)) # update number holding 
        return GDPState(t,aar,h,nam)::State
    end
    return getNextState::Function
end

function getNextStateFunction(p::GDPParams)
    # This function returns the next state function
    function getNextState(s::State,a::Action,rng::AbstractRNG)
        dm = getDelayMatrix(p,s,a) #get delay matrix
        nam = zeros(Flights,p.dInt,p.dInt)
        for i = 1:p.dInt  # put together new arrival demand matrix
            for j = 1:p.dInt
                if i+1 < j && i != p.dInt
                    nam[i,j] = s.am[i+1,j] 
                elseif i+1 == j && i != p.dInt
                    nam[i,j] = s.am[i+1,j] - dm[i+1,j] 
                elseif i+1 > j && i != p.dInt
                    nam[i,j] = s.am[i+1,j] + dm[i,j] - dm[i+1,j] 
                else
                    nam[i,j] = p.sm[s.t+1+p.dInt,s.t+1+p.dInt-j] + dm[p.dInt,j] 
                end
            end
        end
        t = int16(s.t+1) # update time interval
        aar = p.getAAR(int64(s.aar),s.t+1,1,rng) # update aar
        h = int16(max(sum(s.am[1,:])+p.sm[t,t]+sum(p.sm[t,1:t-p.dInt-1])-dm[1,1]+s.h-aar,0)) # update number holding 
        return GDPState(t,aar,h,nam)::State
    end
    return getNextState::Function
end

function getActionFunction(p::GDPParams)
    # This function returns the getAction function, which is the default policy- to not delay any flights
    function getAction(s::State,rng::AbstractRNG)
        aars = s.aar*ones(Int16,p.dInt)
        return rateToAction(p,s,aars)::Action
    end 
    return getAction::Function
end

function getDelayMatrix(p::GDPParams,s::State,a::Action)
    # This function returns the delay matrix, a (dInt x dInt) matrix whose (i,j) entry is the number of delays to be assigned to the (i,j) entry of the arrival demand matrix. 
    # dOpt (the delay option) controls the way in which delays assigned in an interval are distributed across flights scheduled to depart in the next 1...k intervals
    # dOpt = shortestfirst: Delays shortest flights first (i.e. first delays Flights departing in the next interval, then 2,3...k)
    # dOpt = random: Randomly delays flights in departing intervals using numbers drawn from a uniform distribution
    dm = zeros(Flights,p.dInt,p.dInt) 
    tdmax = zeros(Flights,p.dInt) # vector with dInt entries containing max number of Flights that can be delayed in the corresponding interval 
    for i = 1:p.dInt
        tdmax[i] = sum(s.am[i,1:i]) 
    end
    if p.dOpt == :shortestfirst
        for i = 1:p.dInt
            td = ifloor(min(a.td[i],tdmax[i])) 
            j = 1
            while td > 0 && j <= p.dInt
                if s.am[i,j] == 0 || dm[i,j] == s.am[i,j]
                    j += 1
                else
                    dm[i,j] += one(Int16)
                    td -= 1 
                end
            end
        end
    elseif p.dOpt == :longestfirst
        for i = 1:p.dInt
            td = ifloor(min(a.td[i],tdmax[i])) 
            j = i
            while td > 0 && j > 0
                if s.am[i,j] == 0 || dm[i,j] == s.am[i,j]
                    j -= 1
                else
                    dm[i,j] += one(Int16) 
                    td -= 1 
                end
            end
        end
    elseif p.dOpt == :random
        for i = 1:p.dInt
            td = min(a.td[i],tdmax[i])
            dvec = zeros(Int,p.dInt)
            cnt = 0
            while true
                if cnt == td
                    break
                end
                rn = rand(1:i)
                if s.am[i,rn] > dvec[rn]
                    dvec[rn] += 1
                    cnt += 1
                end
            end
            for j = 1:length(dvec)
                dm[i,j] = dvec[j]
            end
        end
    else error("Not a valid option")
    end
    return dm::Array{Flights,2}
end

function getPossibleActionsFunction(p::GDPParams)
    # This function returns the function that returns the possible actions given the PAARs and schedule
    function getPossibleActions(s::State)
        a = Action{}
        actions = Action[] 
        tmp = collect(product(repeated(p.aars,int64(p.dInt))...))
        for i = 1:length(tmp)
            a = rateToAction(p,s,[tmp[i]...])
            if !in(a,actions)
                push!(actions,a)
            end
        end
        return actions::Array{Action,1}
    end
    return getPossibleActions::Function
end

function rateToAction(p::GDPParams,s::State,paars::Array{Flights,1})
    # This function converts PAARs to delays 
    td = zeros(Flights,1,p.dInt)
    for i = 1:p.dInt
        if i == 1
            td[i] = max(sum(s.am[i,:])+sum(p.sm[s.t+i,1:s.t+i-p.dInt-1])+p.sm[s.t+i,s.t+i]+s.h-paars[i],zero(Int16))
        else
            td[i] = max(sum(s.am[i,:])+sum(p.sm[s.t+i,1:s.t+i-p.dInt-1])+p.sm[s.t+i,s.t+i]-paars[i]+td[i-1],zero(Int16))
        end
    end 
    a = GDPAction(td)
    return a::Action
end

end # module
