"""
    KwargsMapping(; flux_function = :notflux, field_names = (), flux_kwargs = (),
                    field_types = (), def_values = ())

Define a mapping of keyword arguments to interface a blueprint with a `Flux` function or
constructor.
"""
struct KwargsMapping{N,T1<:NTuple{N,Symbol},T2<:NTuple{N,Union{Type,Symbol}},
                     T3<:NTuple{N,Any}}
    flux_function::Symbol
    field_names::T1
    flux_kwargs::T1
    field_types::T2
    def_values::T3
end

function KwargsMapping(; flux_function = :notflux, field_names = (), flux_kwargs = (),
                         field_types = (), def_values = ())
    KwargsMapping(flux_function, field_names, flux_kwargs, field_types, def_values)
end

@inline eachkwargs(km::KwargsMapping) = zip(km.field_names, km.flux_kwargs, km.field_types,
                                    km.def_values)
"""
    cyanotype(bp::AbstractBlueprint; kwargs...)

Create a new blueprint from `bp` with the modifications defined by `kwargs`. This method is
automatically generated by the `@cyanotype` macro during the process of defining a
blueprint.
"""
cyanotype

"""
    @cyanotype(doc, expr)
    @cyanotype(doc, kmap, expr)

Defines a blueprint `DataType` with documentation `doc` and a struct declaration defined in
`expr`. If the blueprint directly refers to a `Flux` function or constructor, `kmap` is the
name of [`keyword arguments mapping`](@see KwargsMapping). If there is some fields
documentation in `expr`, it is automatically appended to `doc`.

Automatically generated functions:
* `FooBluePrint(; kwargs...)`: keyword argument constructor for `FooBluePrint`
* `mapping(::FooBluePrint)`: return, if defined, the mapping `kmap`
* [`cyanotype(bp::FooBluePrint)`](@ref cyanotype)
* `kwargs(bp::FooBluePrint)`: return, if defined, a `Dict` with the keyword arguments for
the `Flux` function or constructor, it can be used as follow:
`flux_function(arg1, arg2; kwargs(bp)...)`

# Example:

```julia
using Cyanotype: @cyanotype

@cyanotype (
\"""
A FooBlueprint as example.
\"""
) (
struct FooBlueprint{A<:Function}
    \"""`activation`: activation function\"""
    activation::A = relu
end
)
```

For the keyword arguments mapping usage, see [`KwargsMapping`](@see KwargsMapping)
documentation.
"""
macro cyanotype(doc, expr)
    expr = macroexpand(__module__, expr)
    esc(_cyanotype(__module__, doc, :(KwargsMapping()), expr.args[2], expr.args[3]))
end

macro cyanotype(doc, kmap, expr)
    expr = macroexpand(__module__, expr)
    esc(_cyanotype(__module__, doc, kmap, expr.args[2], expr.args[3]))
end

############################################################################################
#                                   INTERNAL FUNCTIONS                                     #
############################################################################################

function _cyanotype(mod, doc, kmexp, head, body)
    # Forces the struct to inherit from AbstractCyano
    if head isa Symbol || head.head === :curly
        # It is not type stable to do that, since the head type is changed, but at this
        # stage performance is not a big deal.
        head = Expr(:<:, head, Cyanotype.AbstractBlueprint)
    end

    kmap = eval(kmexp)

    # Flux name
    flname = kmap.flux_function

    # Blueprint name
    name = _struct_name(head)

    # Field names
    fnames = Symbol[]

    # Field declarations
    fields = Union{Symbol,Expr}[]

    # Constructor keyword arguments
    kwargs = Union{Symbol,Expr}[]

    # Field documentations
    fdocs = String[]

    # Extracts fields from the body
    tmpdoc = ""
    for f in body.args
        if f isa LineNumberNode
            continue
        elseif f isa String
            tmpdoc = f
        #elseif f isa Expr && f.head === :string
        #    tmpdoc = eval(f.args[1])
        #    @warn("beurk")
        # Field auto-generated by a macro
        elseif f isa Expr && f.head === :block && f.args[1] isa LineNumberNode
            _push_field!(fields, kwargs, fnames, fdocs, _parse_body_field(f.args[2])...)
        else
            _push_field!(fields, kwargs, fnames, fdocs, _parse_body_field(f, tmpdoc)...)
            tmpdoc = ""
        end
    end

    # Adds fields defined by kmap
    for (fname, flarg, T, def) in eachkwargs(kmap)
        fdoc = "`$fname`: see [`$flarg`](@ref ) (default `$def`)" #Flux.$flname
        _push_field!(fields, kwargs, fnames, fdocs, fname, T, def, fdoc)
    end

    # Generates the struct and its associated functions
    quote
        """$($(_generate_documentation(fdocs, doc)))"""
        struct $head
            $(fields...)
        end
        $(_kwargs_constructor(name, fnames, kwargs))
        $(_mapping_func1(mod, name, kmap))
        $(_getfields_func(mod, name, fnames))
        $(_cyanotype_func(mod, name))
        $(_kwargs_func(mod, name))
    end
end

function _struct_name(head)
    #if head isa Symbol
    #    sname = head
    #elseif head.head === :<:
        # No parametric type
        if head.args[1] isa Symbol
            sname = head.args[1]
        else
            sname = head.args[1].args[1]
        end
    #else
    #    sname = head.args[1]
    #end
end

_parse_body_field(field::Symbol, doc = "") = field, :Any, nothing, doc

function _parse_body_field(field::Expr, doc = "")
    # With default value
    if field.head === :(=)
        def = field.args[2]
        # No type annotation
        if field.args[1] isa Symbol
            name = field.args[1]
            T = :Any
        # With type annotation
        else
            name = field.args[1].args[1]
            T = field.args[1].args[2]
        end
    # With type annotation and default value
    elseif field.head === :(::)
        def = nothing
        name = field.args[1]
        T = field.args[2]
    # Block
    # It seems this part is dead code, but I keep it for now
    elseif field.head === :block
        # println(dump(field))
        # Documentation block
        if field.args[2].head === :call && field.args[2].args[1] === Base.Docs.doc!
            doc = field.args[2].args[4].args[2].args[2]
            name, T, def = _parse_body_field(field.args[1])
        # Other kind of block
        else
            error("Blocks are not allowed in body definition.")
        end
    end
    if isempty(doc)
        doc = "`$name` is not documented"
    end
    name, T, def, doc
end

function _push_field!(fields, kwargs, fnames, fdocs, name, T, def, doc)
    if T === :Any
        push!(fields, :($name))
    else
        push!(fields, :($name::$T))
    end
    if isnothing(def)
        push!(kwargs, :($name))
    else
        push!(kwargs, Expr(:kw, name, :($name=$def)))
    end
    push!(fnames, name)
    push!(fdocs, doc)
end

function _generate_documentation(fdocs, doc)
    docs = """
    $doc

    Keyword arguments:\n
    """
    for d in fdocs
        docs = "$docs - $d\n"
    end
    docs
end

function _kwargs_constructor(name, fnames, kwargs)
    # If there is no field in the struct, no kwargs constructor
    if isempty(fnames) && isempty(kwargs)
        :()
    else
        kw = Expr(:parameters, kwargs...)
        :($name($kw) = $name($(fnames...)))
    end
end

function _mapping_func1(mod, name, kmap)
    func = mod === Cyanotype ? :(mapping) : :(Cyanotype.mapping)
    :($func(::$name) = $(kmap))
end

function _getfields_func(mod, name, fnames)
    func = mod === Cyanotype ? :(getfields) : :(Cyanotype.getfields)
    gf = [:($f = cya.$f) for f in fnames]
    :($func(cya::$(name)) = (; $(gf...),))
end

function _cyanotype_func(mod, name)
    func = mod === Cyanotype ? :(cyanotype) : :(Cyanotype.cyanotype)
    gf = mod === Cyanotype ? :(getfields) : :(Cyanotype.getfields)
    quote
        function $func(bp::$name; kwargs...)
            args = []
            fields = $gf(bp)
            # Should we check kwarge validity here?
            for k in keys(fields)
                if haskey(kwargs, k)
                    push!(args, kwargs[k])
                else
                    push!(args, fields[k])
                end
            end
            $name(args...)
        end
    end
end

function _kwargs_func(mod, name)
    func = mod === Cyanotype ? :(kwargs) : :(Cyanotype.kwargs)
    mp = mod === Cyanotype ? :(mapping) : :(Cyanotype.mapping)
    gf = mod === Cyanotype ? :(getfields) : :(Cyanotype.getfields)
    quote
        function $func(bp::$name)
            kmap = $mp(bp)
            kwargs = Dict()
            fields = $gf(bp)
            for (i, arg) in enumerate(kmap.flux_kwargs)
                push!(kwargs, arg=>fields[kmap.field_names[i]])
            end
            kwargs
        end
    end
end
