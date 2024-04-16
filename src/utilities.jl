"""
    spread(bp; kwargs...)

Returns a new blueprint with `kwargs` spreaded over all fields of `bp`. This function allows
to modify all nested cyanotypes at once.

```julia
# Nice printing of large objects
using GarishPrint

# Create a blueprint for a Hybrid A-trou Convolution module
hac = HybridAtrouConvBp();
# By default activation functions are relu
pprint(hac)

# Create a blueprint for a double convolution module with the second convolution as a usual
# convolutionnal layer
conv2 = DoubleConvBp(; convolution1 = hac, convolution2 = ConvBp());
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

function Base.replace(bp, fieldname, old_new)
    old, new = old_new
    stack = _parse_blueprint!([], :top, bp, fieldname, old, new)
    _blueprint_gen(stack)
end

"""
    chcat(x...)

Concatenates the image data along the dimension corresponding to the channels. Image data
should be stored in WHCN order (width, height, channels, batch) or WHDCN (width, height,
depth, channels, batch) in 3D context. Channels are assumed to be the penultimate dimension.

# Example
```julia
julia> x1 = rand(32, 32, 4, 6); # a batch of 6 images (32x32) with 4 channels

julia> x2 = rand(32, 32, 4, 6); # a batch of 6 images (32x32) with 4 channels

julia> chcat(x1, x2) |> size
(32, 32, 8, 6)
```
"""
chcat(x...) = cat(x...; dims = ndims(x[1]) - 1)

chsoftmax(x) = softmax(x; dims = ndims(x) - 1)

ChainRules.@non_differentiable chsoftmax(x)

chmeanpool(x) = mean(x; dims = ndims(x) - 1)

ChainRules.@non_differentiable chmeanpool(x)

chmaxpool(x) = maximum(x; dims = ndims(x) - 1)

ChainRules.@non_differentiable chmaxpool(x)

function flatten_layers(layers...)
    result = []
    _flatten_layers!(result, layers)
    result
end

@inline genk(k, vol) = vol ? (k, k, k) : (k, k)

macro volume()
    esc(
    quote
        """
        `volume`: indicates a building process for three-dimensionnal data (default `false`)
        """
        volume::Bool = false
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

macro activation2(func)
    ref = "[`$func`](@ref $func)"
    doc = "`activation`: activation function (default [`$func`](@ref Flux.$func))"
    esc(
    quote
        """
        $($(doc))
        """
        activation = $func
    end)
end

macro volume2()
    esc(
    quote
        """
        `volume`: indicates a building process for three-dimensionnal data (default `false`)
        """
        volume = false
    end)
end

########################################################################################################################
#                                               INTERNAL FUNCTIONS                                                     #
########################################################################################################################

# Shortcut that do nothing if not an AbstractBlueprint
_parse_blueprint!(::Any, ::Any, ::Any; kwargs...) = nothing
_parse_blueprint!(::Any, ::Any, ::Any, ::Any, ::Any, ::Any) = nothing

# Parses a AbstractBlueprint bp and returns a stack
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

# Parses a AbstractBlueprint bp and returns a stack
function _parse_blueprint!(stack, name, bp::AbstractBlueprint, fieldname, old, new)
    fields = Dict(pairs(getfields(bp)))
    # stack records the AbstractBlueprint object, its fields and name
    push!(stack, (bp, fields, name))
    # Replace field according to old => new
    if haskey(fields, fieldname) && fields[fieldname] == old
        fields[fieldname] = new
    end
    # Reccursive parsing of each field
    for (n, f) in fields
        _parse_blueprint!(stack, n, f, fieldname, old, new)
    end
    stack
end

# Generate a new blueprint from a stack
function _blueprint_gen(stack)
    # Stores blueprints generated from the stack
    blueprints = Dict()
    # Evaluates stack from bottom to top
    for (bp, kw, n) in reverse(stack)
        for k in keys(kw)
            # Modify kw if k is in blueprints dictionnary and is actually an AbstractBlueprint
            if haskey(blueprints, k) && kw[k] isa AbstractBlueprint
                # Consume this blueprint
                #delete!(kw, k)
                #println(kw[k])
                kw[k] = blueprints[k]
                #println("pouet 2")
                delete!(blueprints, k)
                #println("pouet 3")
            end
        end
        # Generates a new blueprint from kw and store it for the further iterations
        blueprints[n] = cyanotype(bp; kw...)
    end
    blueprints[:top]
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
