module Ball

# Solves Single Airport Ground Holding Problem for a single airport given the arrival schedule, a set of capacity scenarios and their respective probabilites. Documentation for this model can be found in Ball et al. 2001

export BallParams, simulate

using JuMP, Gurobi, GDP

type BallParams
    S::Array{Int64}
    M::Array{Int64}
    p::Array{Float64}
    ca::Float64
end

function simulate(p::BallParams)

    # S - (T x 2) vector whose first column is the number scheduled to arrive in each of T intervals not exempt from being assigned ground delay and whose second column is the number scheduled to arrive exempt from being assigned ground delay
    # M - (Q x T) matrix whose (q,i) entry is the landing capacity under scenario q in interval i
    # p - (Q x 1) vector whose i entry is the probability associated with the capacity scenario in the (q,:) entry of M 

    # Cost of ground delay, number of time intervals, number of capacity scenarios
    cg = 1
    T = size(p.S,1)
    Q = length(p.p)

    # Unexempted (D)  and exempted (E) flights
    D = p.S[:,1]
    E = p.S[:,2]

    ball = Model(solver=GurobiSolver(OutputFlag=0))

    # X is a (T x 1) vector whose i entry is the number to ground delay in interval i
    @defVar(ball, X[1:T] >= 0, Int)

    # W is a (Q x T) matrix whose (q,i) entry is the number airborne holding under scenario q in interval i 
    @defVar(ball, W[1:Q,1:T] >= 0, Int) 

    # A is a (T x 1) vector whose i entry is the number rescheduled to arrive in interval i, these can be interpreted as the planned airport acceptance rates
    @defVar(ball, A[1:T] >= 0, Int)

    @defVar(ball, i[1:T] >= 0)
    @defVar(ball, q[1:Q] >= 0)

    @setObjective(ball, Min, cg*sum{X[i],i=2:T-1} + sum{p.p[q]*p.ca*sum{W[q,i],q=1:Q},i=2:T-1})

    @defConstrRef constraint1[1:T]
    @defConstrRef constraint2[1:Q,1:T]
    @defConstrRef constraint3[1:Q]
    @defConstrRef constraint4[1:Q]
    @defConstrRef constraint5[1:2]

    # All flights must be rescheduled to arrive at or later than the current time
    for i = 2:T
        constraint1[i] = @addConstraint(ball, A[i] - X[i-1] + X[i] == D[i])
    end

    # This imposes an upper limit on the number of flights that can land in a given capacity scenario
    for q = 1:Q
        for i = 2:T
            constraint2[q,i] = @addConstraint(ball, A[i] + E[i] + W[q,i-1] - W[q,i] <= p.M[q,i])
        end
    end

    # No flights can be holding in the initial interval
    for q = 1:Q
        constraint3[q] = @addConstraint(ball, W[q,1] == 0)
    end 

    # No flights can be holding in the final interval
    for q = 1:Q
        constraint4[q] = @addConstraint(ball, W[q,T] == 0)
    end 

    # No flight can be delayed in the initial and final interval
    constraint5[1] = @addConstraint(ball, X[1] == 0)
    constraint5[2] = @addConstraint(ball, X[T] == 0)

    status = solve(ball)

    delayed = getValue(X)

    return GDPAction(int16(delayed[:])')::GDPAction

end

end # module
