module DataTools

export getForecast, getSchedule

using DataFrames, HDF5, JLD

function getForecast(airport::Symbol)
    fid = jldopen("../data/forecast.jld","r")
    data = read(fid,string(airport))
    close(fid)
    return data::DataFrame
end

function getSchedule(airport::Symbol,date::ASCIIString)
    df = readtable("../data/"*string(airport)*"_"*date*".csv") # The flight schedule in .csv format
    sm = zeros(Int16,100,100)
    for i = 1:length(df[:Arrival_1])
        try
            if int16(df[:Arrival_1][i]) != zero(Int16)
                sm[int16(df[:Arrival_1][i])+one(Int16),int16(df[:Departure_1][i])+one(Int16)] += int16(df[:Flight][i])
            end
        catch
        end
    end
    return sm::Array{Int16,2}
end

end # module
