"""
    spread(bp; kwargs...)

Create a new blueprint with `kwargs` spreaded over all fields of `bp`. This function allows
to modify all nested blueprint at once.

# Example
```julia
julia> # Create a blueprint for a Hybrid A-trou Convolution unit. By default activation
       # functions are `identity`.

julia> hac = HybridAtrouConvBp();

julia> hac.conv.act |> println
identity

julia> # Create a blueprint for a double convolution unit a the second convolution as a
       # usual convolutionnal layer. By default activation function is `identity`.

julia> conv = DoubleConvBp(; conv1 = hac, conv2 = ConvBp());

julia> # Now let change all activation functions from relu to leakyrelu

julia> bp = spread(conv; act = leakyrelu);

julia> bp.conv1.conv.act |> println
leakyrelu

julia> bp.conv2.act |> println
leakyrelu
```
"""
function spread(bp; kwargs...)
    stack = _parse_blueprint!([], :top, bp; kwargs...)
    _blueprint_gen(stack)
end

"""
    spread(bp, fieldname, old => new)

Create a new blueprint from `bp` for the given `fieldname`, and replace each occurence of
`old` with `new`. This function allows to modify all nested blueprints at once.

# Example
```julia
julia> # Create a blueprint for a Hybrid A-trou Convolution unit. By default activation
       # functions are identity.

julia> hac = HybridAtrouConvBp();

julia> hac.conv.act |> println
identity

julia> # Create a blueprint for a double convolution unit with a second convolution as a
       # usual convolutionnal layer with `relu` as activation.

julia> conv = DoubleConvBp(; conv1 = hac, conv2 = ConvBp(act = relu));

julia> # Now let change all activation functions from relu to leakyrelu

julia> bp = spread(conv, :act, identity => leakyrelu);

julia> bp.conv1.conv.act |> println
leakyrelu

julia> bp.conv2.act |> println
relu
```
"""
function spread(bp, fieldname, old_new)
    old, new = old_new
    stack = _parse_blueprint!([], :top, bp, fieldname, old, new)
    _blueprint_gen(stack)
end

"""
    chcat(x...)

Concatenates the provided data along the channel dimension positioned as the penultimate
dimension.

# Example
```julia
julia> x1 = rand(32, 32, 4, 6); # a batch of 6 images (32x32) with 4 channels

julia> x2 = rand(32, 32, 4, 6); # a batch of 6 images (32x32) with 4 channels

julia> chcat(x1, x2) |> size
(32, 32, 8, 6)
```
"""
chcat(x...) = cat(x...; dims = ndims(x[1]) - 1)

"""
    chsoftmax(x)

Apply a softmax function along the channel dimension. This function is useful for pixel
classification (semantic segmentation).

# Example
```julia
julia> x = rand(2, 2, 3, 1);

julia> sum(chsoftmax(x), dims=3)
2×2×1×1 Array{Float64, 4}:
[:, :, 1, 1] =
 1.0  1.0
 1.0  1.0
```
See also [`PixelClassifierBp`](@ref).
"""
chsoftmax(x) = softmax(x; dims = ndims(x) - 1)

ChainRules.@non_differentiable chsoftmax(x)

"""
    chmeanpool(x)

Pools all channels using the mean function.

# Example
```julia
julia> x = rand(2, 2, 3, 1);

julia> chmeanpool(x)
2×2×1×1 Array{Float64, 4}:
[:, :, 1, 1] =
 0.64925   0.242134
 0.153542  0.312039
 ```
"""
chmeanpool(x) = mean(x; dims = ndims(x) - 1)

ChainRules.@non_differentiable chmeanpool(x)

"""
    chmaxpool(x)

Pools all channels using the maximum function.

# Example
```julia
julia> x = rand(2, 2, 3, 1);

julia> Cyanotype.chmaxpool(x)
2×2×1×1 Array{Float64, 4}:
[:, :, 1, 1] =
 0.930081  0.648555
 0.759567  0.803717
```
"""
chmaxpool(x) = maximum(x; dims = ndims(x) - 1)

ChainRules.@non_differentiable chmaxpool(x)

"""
    flatten_layers(layers...)

Flatten a nested Vector/Tuple/Chain into a single Vector. If any layers contains identity
function it is skipped.

# Example
```julia
julia> flatten_layers(["a",("b","c",["d","e",identity])])
5-element Vector{Any}:
 "a"
 "b"
 "c"
 "d"
 "e"
```
"""
function flatten_layers(layers...)
    result = []
    _flatten_layers!(result, layers)
    result
end

"""
    genk(k, vol)

Helping function used to generate a kernel tuple (k,k) or (k,k,k) if `vol` is true.
"""
@inline genk(k, vol) = vol ? (k, k, k) : (k, k)

"""
    @activation(func)

Helping macro used to define an activation function within a blueprint definition.
"""
macro activation(func)
    doc = "`act`: activation function (default [`$func`](@ref))"
    esc(
        quote
            """
            $($(doc))
            """
            act = $func
        end
    )
end

"""
@volume

Helping macro to define 'vol' field within a blueprint definition.
"""
macro volume()
    esc(
        quote
            """
            `vol`: indicates a building process for three-dimensionnal data (default
             `false`)
            """
            vol = false
        end
    )
end

############################################################################################
#                                   INTERNAL FUNCTIONS                                     #
############################################################################################

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
                kw[k] = blueprints[k]
                delete!(blueprints, k)
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
