function readParams(airport::Symbol)
    adpm = Float64[]
    if airport == :SFO
        adpm = zeros(2,6,61,61)
    elseif airport == :EWR
        adpm = zeros(2,6,53,53)
    end
    rawData = readcsv("../adpm/params/adpm_params_"*lowercase(string(airport)))
    for i = 1:length(adpm)
        adpm[i] = rawData[i]
    end
    return adpm
end
