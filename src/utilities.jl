"""
    spread(bp; kwargs...)

Return a new cyanotype blueprint with `kwargs` spreaded over all fields of `bp`. This
function allows to modify all nested cyanotypes at once.

```julia
# Nice printing of large objects
using GarishPrint

# Create a blueprint for a Hybrid A-trou Convolution module
hac = CyHybridAtrouConv();
# By default activation functions are relu
pprint(hac)

# Create a blueprint for a double convolution module with the second convolution as a usual
# convolutionnal layer
conv2 = CyDoubleConv(; convolution1 = hac, convolution2 = CyConv());
# By default activation functions are relu
pprint(conv2)

# Now let change all activation functions from relu to leakyrelu
spread(conv2; activation = leakyrelu) |> pprint
```
"""
function spread(bp; kwargs...)
    stack = _parse_blueprint!([], :top, bp; kwargs...)
    _blueprint_gen(stack)
end

# Shortcut that do nothing if not an AbstractBlueprint
_parse_blueprint!(::Any, ::Any, ::Any; kwargs...) = nothing

# Parses a AbstractBlueprint blueprint and returns a stack
function _parse_blueprint!(stack, name, bp::AbstractBlueprint; kwargs...)
    fields = Dict(pairs(getfields(bp)))
    # stack records the AbstractBlueprint object, its fields and name
    push!(stack, (bp, fields, name))
    # Rename fields according to kwargs
    for (k, a) in kwargs
        if haskey(fields, k)
            fields[k] = a
        end
    end
    # Reccursive parsing of each field
    for (n, f) in fields
        _parse_blueprint!(stack, n, f; kwargs...)
    end
    stack
end

# Generate a new blueprint from a stack
function _blueprint_gen(stack)
    # Store blueprints generated from the stack
    blueprints = Dict()
    # Evaluate stack from bottom to top
    for (bp, kw, n) in reverse(stack)
        for k in keys(kw)
            # Modify kw if k is in blueprints dictionnary and is actually an
            # AbstractBlueprint
            if haskey(blueprints, k) && kw[k] isa AbstractBlueprint
                # Consume this blueprint
                kw[k] = blueprints[k]
                delete!(blueprints, k)
            end
        end
        # Generate a new blueprint from kw and store it for the further iterations
        blueprints[n] = cyanotype(bp; kw...)
    end
    blueprints[:top]
end

#str2sym(d) = Dict(Symbol(k) => v for (k,v) in d)

#sym2str(d) = Dict(String(k) => v for (k,v) in d)

function flatten_layers(layers...) #clean_layers
    result = []
    #@code_warntype _flatten_layers!(result, layers)
    _flatten_layers!(result, layers)
    result
end

function _flatten_layers!(buffer, layers)
    if applicable(iterate, layers)
        for l in layers
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

#activation_doc(func = relu) = "`activation`: activation function, by default [`$func`](@ref Flux.$func)"

#const ACTIVATION_DOC_RELU = activation_doc()

function autogen_build_doc(T, with_kernel_size, with_channels)
    doc = "build("
    if with_kernel_size
        doc = doc * "kernel_size, "
    end
    if with_channels
        doc = doc * "channels, "
    end
    doc = doc * "cya::$T)"
    "$doc See [`$T`](@ref)"
end

#const VOLUMETRIC_FIELD = :(volumetric::Bool = false) # pas encore test

macro volumetric()
    esc(
    quote
        """
        `volumetric`: indicates a building process for three-dimensionnal data (default `false`)
        """
        volumetric::Bool = false
    end)
end

macro activation(func)
    ref = "[`$func`](@ref $func)"
    doc = "`activation`: activation function (default [`$func`](@ref Flux.$func))"
    esc(
    quote
        """
        $($(doc))
        """
        activation::A = $func
    end)
end
