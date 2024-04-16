"""
    cyanotype(bp::AbstractBlueprint; kwargs...)

Creates a new blueprint from `bp` with the modifications defined by `kwargs`. This method is
automatically generated by the `@cyanotype` macro during the process of defining a
blueprint.
"""
cyanotype

"""
    KwargsMapping(; flux_function = :notflux, field_names = (), flargs = (),
                    ftypes = (), defval = ())

Defines a mapping of keyword arguments to interface a blueprint with a `Flux` function or
constructor.
"""
struct KwargsMapping
    flfunc
    fnames
    flargs
    defval
end

function KwargsMapping(; flfunc=:notflux, fnames=(), flargs=(), defval=())
    KwargsMapping(flfunc, fnames, flargs, defval)
end

@inline eachkwargs(km::KwargsMapping) = zip(km.fnames, km.flargs, km.defval)

"""
    @cyanotype(expr)
    @cyanotype begin
        [kmap]
        [doc]
        expr
    end

Defines a blueprint `DataType` with documentation `doc` and a struct declaration defined in
`expr`. If the blueprint directly refers to a `Flux` function or constructor, `kmap` is the
name of [`keyword arguments mapping`](@see KwargsMapping). If there is some fields
documentation in `expr`, it is automatically appended to `doc`.

Automatically generated functions:
* `FooBluePrint(; kwargs...)`: keyword argument constructor for `FooBluePrint`
* `mapping(::FooBluePrint)`: return, if defined, the mapping `kmap`
* `cyanotype(bp::FooBluePrint; kwargs...)`
* `kwargs(bp::FooBluePrint)`: return, if defined, a `Dict` with the keyword arguments for
the `Flux` function or constructor, it can be used as follow:
`flux_function(arg1, arg2; kwargs(bp)...)`

# Example:

```julia
using Cyanotype

@cyanotype begin
    \"""
    A FooBlueprint as example.
    \"""
    struct FooBlueprint{A<:Function}
        \"""`act`: activation function\"""
        act::A = relu
    end
end
```

For the keyword arguments mapping usage, see [`KwargsMapping`](@see KwargsMapping)
documentation.
"""
macro cyanotype(expr)
    doc, kmexp, head, body = _parse_expr(__module__, expr)
    esc(_cyanotype(__module__, doc, kmexp, head, body))
end

macro cyanotype(opt, expr)
    cons = true
    if opt.head === :(=) && opt.args[1] === :constructor
        cons = opt.args[2]
    end #else error
    doc, kmexp, head, body = _parse_expr(__module__, expr)
    esc(_cyanotype(__module__, doc, kmexp, head, body, cons))
end

########################################################################################################################
#                                               INTERNAL FUNCTIONS                                                     #
########################################################################################################################

function _parse_expr(mod, expr)
    doc = ""
    kmexp = :(KwargsMapping())
    head = :()
    body = :()
    for arg in expr.args
        if arg isa Expr
            if arg.head === :call && arg.args[1] === :KwargsMapping
                kmexp = arg
            elseif arg.head === :macrocall
                for a in arg.args
                    if a isa String
                        doc = a
                    elseif a isa Expr && a.head === :struct
                        tmp = macroexpand(mod, a)
                        head = tmp.args[2]
                        body = tmp.args[3]
                    end
                end
            elseif arg.head === :struct
                tmp = macroexpand(mod, arg)
                head = tmp.args[2]
                body = tmp.args[3]
            end
        end
    end
    doc, kmexp, head, body
end

function _cyanotype(mod, doc, kmexp, head, body, cons = true)
    # Forces the struct to inherit from AbstractCyano
    if head isa Symbol || head.head === :curly
        # It is not type stable to do that, since the head type is changed, but at this stage performance is not a big
        # deal.
        head = Expr(:<:, head, Cyanotype.AbstractBlueprint)
    end

    kmap = Core.eval(mod, kmexp) #eval(kmexp)

    # Flux name
    flname = kmap.flfunc

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
    for (fname, flarg, def) in eachkwargs(kmap) #, T
        fdoc = "`$fname`: see [`$flarg`](@ref ) (default `$def`)"
        _push_field!(fields, kwargs, fnames, fdocs, fname, Any, def, fdoc)
    end

    # Generates the struct and its associated functions
    quote
        """$($(_generate_documentation(fdocs, doc, cons)))"""
        struct $head
            $(fields...)
        end
        $(_kwargs_constructor(name, fnames, kwargs, cons))
        $(_mapping_func1(mod, name, kmap))
        $(_getfields_func(mod, name, fnames))
        $(_cyanotype_func(mod, name))
        $(_kwargs_func(mod, name))
        $(_show_func(name))
    end
end

function _struct_name(head)
    if head.args[1] isa Symbol
        sname = head.args[1]
    else
        sname = head.args[1].args[1]
    end
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

function _generate_documentation(fdocs, doc, cons)
    docs = doc
    if cons
        docs *= "\nKeyword arguments:\n"
        for d in fdocs
            docs = "$docs - $d\n"
        end
    end
    docs
end

function _kwargs_constructor(name, fnames, kwargs, cons)
    # If there is no field in the struct, no kwargs constructor
    if isempty(fnames) && isempty(kwargs) || !cons
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
            for (i, arg) in enumerate(kmap.flargs)
                push!(kwargs, arg=>fields[kmap.fnames[i]])
            end
            kwargs
        end
    end
end

function _show_func(name)
    quote
        function Base.show(io::IO, bp::$name)
            dump(
                IOContext(io, :limit => true, :compact => true, :color => true),
                bp,
                maxdepth=1
            )
        end
    end
end
