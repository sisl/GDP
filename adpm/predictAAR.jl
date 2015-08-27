function mc(df::DataFrame,ind::Int)
    # This function categorizes the weather conditions as VMC (2) or IMC (1)
    res = 1.
    if df[symbol("Ceiling_"*string(ind))][1]*100.0 > 3000. && df[symbol("Visibility_"*string(ind))][1] > 5.
        res = 2.
    end
    return res::Float64
end

function predictAAR(model::Array,forecast::DataFrame) 
    # This function returns the getAAR function, a model for random generation of AARs
    nAARs = size(model,4)
    function getAAR(previousAAR::Int,forecastTime::Int,leadTime::Int,rng::AbstractRNG)
        conditional = reshape(model[mc(forecast[forecastTime,names(forecast)],leadTime),1,previousAAR+1,:],nAARs,1)
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
