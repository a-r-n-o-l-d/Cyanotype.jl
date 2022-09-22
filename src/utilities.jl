str2sym(d) = Dict(Symbol(k) => v for (k,v) in d)

sym2str(d) = Dict(String(k) => v for (k,v) in d)

function flatten_layers(layers...)
    result = []
    #@code_warntype _flatten_layers!(result, layers)
    _flatten_layers!(result, layers)
    result
end

function _flatten_layers!(buffer, layers)
    if applicable(iterate, layers)
        for l ∈ layers
            if l isa AbstractVector || l isa Chain || l isa Tuple
                #@code_warntype _flatten_layers!(buffer, l)
                _flatten_layers!(buffer, l)
            elseif l !== Flux.identity
                push!(buffer, l)
            end
        end
    else
        push!(buffer, layers)
    end
end
