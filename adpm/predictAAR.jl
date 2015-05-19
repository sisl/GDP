function mc(df::DataFrame,ind::Int)
    # This function categorizes the weather conditions as VMC (2) or IMC (1)
    res = 1.
    if df[:Ceiling_0][ind]*100.0 > 3000. && df[:Visibility_0][ind] > 5.
        res = 2.
    end
    return res::Float64
end

function predictAAR(model::Array,forecast::DataFrame) 
    # This function returns the getAAR function, a model for random generation of AARs
    nAARs = size(model,4)
    function getAAR(previousAAR,forecastTime,rng::AbstractRNG)
        t = 1 # always use the forecast made at time t-1
        conditional = reshape(model[mc(forecast[forecastTime,names(forecast)],t),t,previousAAR+1,:],nAARs,1)
        rn = rand(rng)
        cnt = 0.
        for i = 1:nAARs
            cnt += conditional[i]
            if cnt > rn
                return i-1
            end
        end
    end
    return getAAR::Function
end
