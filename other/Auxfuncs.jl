module Auxfuncs

# This module adds the functionality to include user-defined types as keys in maps

import Base.hash, Base.isequal, GDP.GDPAction

function hash(a::GDPAction)
    h::Uint = hash(size(a.td))+1
    for i=1:length(a.td)
        h = bitmix(h,int(hash(a.td[i])))
    end
    return h
end

function isequal(w::GDPAction,v::GDPAction) 
    return isequal(w.td,v.td)
end

end # module
