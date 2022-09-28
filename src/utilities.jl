#str2sym(d) = Dict(Symbol(k) => v for (k,v) in d)

#sym2str(d) = Dict(String(k) => v for (k,v) in d)

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

activation_doc(func = relu) = "`activation`: activation function, by default [`$func`](@ref Flux.$func)"

const ACTIVATION_DOC_RELU = activation_doc()

function autogen_build_doc(T, with_kernel_size, with_channels)
    doc = "build("
    if with_kernel_size
        doc = doc * "kernel_size, "
    end
    if with_channels
        doc = doc * "channels, "
    end
    doc = doc * "cya::$T)"
    "
        $doc
    "
end

const VOLUMETRIC_FIELD = :(volumetric::Bool = false)
