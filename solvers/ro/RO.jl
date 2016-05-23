module RO

# Solves the Single Airport Ground Holding Problem given the arrival schedule, a set of capacity scenarios, and their respective probabilities. Documentation for this model can be found in Richetta and Odoni 1994. This module postprocesses the decision variables to return the PAARs

export ROParams, simulate

using JuMP, Gurobi, GDP

type ROParams
    ts::Array{Int64}
    S::Array{Int64}
    M::Array{Int64}
    p::Array{Float64}
    ca::Float64
end

function couplingConstraints(ro::Model,X,T::Int,p::ROParams,oldInds::Array{Int},s::Int)
    # Add coupling constraints for connected branches

    # Base cases
    if length(oldInds) < 2 || s > length(p.ts)
        return
    end

    # Find unique capacities
    capacities = unique(p.M[oldInds,p.ts[s]])

    for c in capacities
        inds = find(p.M[oldInds,p.ts[s]].==c)
        for k = 1:length(inds)-1
            for i = 1:T+1
                for j = 1:T+1
                    @constraint(ro, X[inds[k],s,i,j] == X[inds[k]+1,s,i,j])
                end
            end
        end
        couplingConstraints(ro,X,T,p,inds,s+1)
    end
end


function simulate(p::ROParams)

    # ts - (S x 1) vector whose i entry is the start interval of the ith stage

    # S - (Q x T+1) matrix whose (s, i) entry is the number scheduled to depart during stage s and arrive during interval i

    # M - (Q x T+1) matrix whose (q, i) entry is the landing capacity under scenario q in interval i

    # p - (Q x 1) vector whose i entry is the marginal probability associated with the capacity scenario in the (q, :) entry of M 

    # Cost of ground delay, number of time intervals, number of capacity scenarios
    cg = 1
    T = size(p.M,2)-1 
    S = length(p.ts)
    Q = length(p.p)

    ro = Model(solver=GurobiSolver(OutputFlag=0,Threads=1))

    # X is a (Q x Q x T+2 x T+2) matrix whose (q, s, i, j) entry is the number of aircraft scheduled to depart during stage s and arrive at interval i rescheduled to arrive at interval j under scenario q
    @variable(ro, X[1:Q, 1:S, 0:T+1, 0:T+1] >= 0, Int)

    # W is a (Q x T+2) matrix whose (q, i) entry is the number absorbing air delay under scenario q in interval i 
    @variable(ro, W[1:Q, 0:T+1] >= 0, Int)

    # Indices are all non-negative
    @variable(ro, q[1:Q] >= 0)
    @variable(ro, s[1:Q] >= 0)
    @variable(ro, i[0:T+1] >= 0)
    @variable(ro, j[0:T+1] >= 0)

    # Objective function -  minimize a linear combination of ground and air delay
    @objective(ro, Min, sum{p.p[q]*(cg*sum{sum{sum{(j-i)*X[q,s,i,j],j=i+1:T+1},i=p.ts[s]+1:T},s=1:S}+p.ca*sum{W[q,i],i=1:T+1}),q=1:Q}) 

    # Constraints 1-3 are detailed on page 173 of the journal article

    for q = 1:Q
        for s = 1:S
            for i = p.ts[s]+1:T
                @constraint(ro, sum{X[q,s,i,j],j=i:T+1} == p.S[s,i])
            end
        end
    end

    for q = 1:Q
        for i = 1:T+1
            temp = 0.
            for s = 1:S
                for j = p.ts[s]+1:i
                    if p.ts[s] < i
                        temp += X[q,s,j,i] 
                    end
                end
            end
            @constraint(ro, -W[q,i] + W[q,i-1] + temp <= p.M[q,i])
        end
    end

    for q = 1:Q
        @constraint(ro, W[q,0] == 0)
    end

    for q = 1:Q
        @constraint(ro, W[q,T+1] == 0)
    end

    # All scenarios must be connected at the root of the tree
    for q = 1:Q-1
        for i = 1:T+1
            for j = 1:T+1
                for s = 1:2
                    @constraint(ro, X[q,s,i,j] == X[q+1,s,i,j])
                end
            end
        end
    end

    # Add coupling constraints for connected branches
    couplingConstraints(ro,X,T,p,collect(1:Q),3)

    status = solve(ro)

    delayed = getvalue(X)

    a = GDPAction(zeros(Int16,T,1)) # 1 empty interval

    for i = 2:length(a.td)+1
        for j = i+1:T+1
            for s = 1:2
                a.td[i-1] += delayed[1,s,i,j]
            end
        end
    end

    for i = 2:length(a.td)+1
        for k = 2:i-1
            for j = i+1:T+1
                for s = 1:2
                    a.td[i-1] += delayed[1,s,k,j]
                end
            end
        end
    end

    return a::GDPAction

end

end # module
