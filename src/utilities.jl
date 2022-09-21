str2sym(d) = Dict(Symbol(k) => v for (k,v) in d)

sym2str(d) = Dict(String(k) => v for (k,v) in d)
