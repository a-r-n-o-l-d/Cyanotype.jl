abstract type AbstractCyanotype end

struct KwargsMapping{N,T1<:NTuple{N,Symbol},T2<:NTuple{N,Union{Type,Symbol}},T3<:NTuple{N,Any}} #
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

MAPPINGS::Dict{Symbol,KwargsMapping} = Dict(:empty_map => KwargsMapping())

function register_mapping!(mapping)
    haskey(MAPPINGS, first(mapping)) && @warn "Map $(first(mapping)) already exists in Cyanotype.MAPPINGS."
    push!(MAPPINGS, mapping)
end


#=def(x) = :(() -> $x)

struct CyFunc #{F<:Function}
    func #::F
end
func(x) = x
func(cf::CyFunc) = cf.func

def(x::Function) = :(CyFunc($x))=#

macro cyanotype(doc, expr)
    expr = macroexpand(__module__, expr)
    esc(_cyanotype(__module__, doc, :empty_map, expr.args[2], expr.args[3]))
end

macro cyanotype(kmap, doc, expr)
    expr = macroexpand(__module__, expr)
    esc(_cyanotype(__module__, doc, kmap, expr.args[2], expr.args[3]))
end

############################################################################################
#                                   INTERNAL FUNCTIONS                                     #
############################################################################################

function _cyanotype(mod, doc, kmap, head, body)
    # Forces the struct to inherit from AbstractCyano
    if head isa Symbol || head.head === :curly
        # It is not type stable to do that, since the head type is changed, but at this
        # stage performance is not a big deal.
        head = Expr(:<:, head, Cyanotype.AbstractCyanotype)
    end

    # Flux name
    flname = Cyanotype.MAPPINGS[kmap].flux_function #

    # Flux ref for documentation
    #flref = "[`$flname`](@ref Flux.$flname)"

    # Cyanotype name
    cyname = _struct_name(head)

    #cydoc = eval(macroexpand(mod, :(@doc $cyname)))
    #println(cydoc)

    # Field names
    fnames = Symbol[]

    # Flux names
    #flnames = Symbol[]

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
        elseif f isa Expr && f.head === :string
            #println(f.args)
            tmpdoc = eval(f.args[1]) # beurk
            @warn("beurk")
        # Field auto-generated by a macro
        elseif f isa Expr && f.head === :block && f.args[1] isa LineNumberNode
            #macroexpand(mod, f) |> println
            #println(f.args)
            #_parse_body_field(f.args[2]) |> println
            _push_field!(fields, kwargs, fnames, fdocs, _parse_body_field(f.args[2])...)
            #eval(f.args[2].args[2]) |> println
            #_parse_body_field(f.args[2].args[1], f.args[2].args[2].args[4].args[2].args[2]) |> println
        else
            _push_field!(fields, kwargs, fnames, fdocs, _parse_body_field(f, tmpdoc)...)
            tmpdoc = ""
        end
    end

    # Adds fields defined by kmap
    for (name, flarg, T, def) in eachkwargs(Cyanotype.MAPPINGS[kmap]) #Cyanotype.
        fdoc = "`$name`: see [`$flarg`](@ref Flux.$flname) (default `$def`)"
        _push_field!(fields, kwargs, fnames, fdocs, name, T, def, fdoc)
        #push!(flnames, flname)
    end

    # Generates the struct and its associated functions
    quote
        """$($(_generate_documentation(cyname, fdocs, doc)))"""
        struct $head
            $(fields...)
        end
        $(_kwargs_constructor(cyname, fnames, kwargs))
        $(_mapping_func1(mod, cyname, kmap))
        $(_getfields_func(mod, cyname, fnames))
        $(_cyanotype_func(mod, cyname))
        $(_kwargs_func(mod, cyname))
        #$(_getfields_func(cyaname, fnames))
        #$(_getfluxfields_func(cyaname, fnames, flnames))
        #=$(_copy_constructor(cyaname))
        $(_new_func(cyaname))

        $(_mapping_func2(cyaname, kmap))
        $(_getfields_func(cyaname, fnames))
        $(_curate_func(cyaname))=#
    end
end

function _struct_name(head)
    if head isa Symbol
        sname = head
    elseif head.head === :<:
        # No parametric type
        if head.args[1] isa Symbol
            sname = head.args[1]
        else
            sname = head.args[1].args[1]
        end
    else
        sname = head.args[1]
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
            #println(_parse_body_field(field.args[1]))
            #println(dump(field))
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

function _generate_documentation(cyname, fdocs, doc)
    #$cyname(; kwargs...)
    docs = """$doc

    Keyword arguments:\n
    """
    for d in fdocs
        docs = "$docs - $d\n"
    end
    docs
end

function _kwargs_constructor(cyname, fnames, kwargs)
    # If there is no field in the struct, no kwargs constructor
    if isempty(fnames) && isempty(kwargs)
        :()
    else
        kw = Expr(:parameters, kwargs...)
        :($cyname($kw) = $cyname($(fnames...)))
    end
end

function _mapping_func1(mod, cyname, kmap)
    func = mod === Cyanotype ? :(mapping) : :(Cyanotype.mapping)
    :($func(::$cyname) = $(Cyanotype.MAPPINGS[kmap]))
end

function _getfields_func(mod, cyname, fnames)
    func = mod === Cyanotype ? :(getfields) : :(Cyanotype.getfields)
    gf = [:($f = cya.$f) for f in fnames]
    :($func(cya::$(cyname)) = (; $(gf...),))
end

function _cyanotype_func(mod, cyname)
    func = mod === Cyanotype ? :(cyanotype) : :(Cyanotype.cyanotype)
    quote
        function $func(cy::$cyname; kwargs...)#::$cyname #Cyanotype.cyanotype si en dehors du module
            args = []
            fields = getfields(cy)
            # verifier la validite des kwargs ?
            for k in keys(fields)
                if haskey(kwargs, k)
                    push!(args, kwargs[k])
                else
                    push!(args, fields[k])
                end
            end
            $cyname(args...)
        end
    end
end

function _kwargs_func(mod, cyname)
    func = mod === Cyanotype ? :(kwargs) : :(Cyanotype.kwargs)
    quote
        function $func(cy::$cyname)
            kmap = mapping(cy)
            kwargs = Dict()
            fields = getfields(cy)
            for (i, arg) in enumerate(kmap.flux_kwargs)
                push!(kwargs, arg=>fields[kmap.field_names[i]])
                #push!(kwargs, arg=>getfield(cy, kmap.field_names[i]))
            end
            kwargs
        end
    end
end

#=
function _getfields_func(sname, fnames)
    gf = [:($f = cya.$f) for f in fnames]
    #kwf = Expr(:parameters, gf...)
    :(getfields(cya::$(sname)) = (; $(gf...),))
end


function _getfluxfields_func(sname, fnames, flnames)
    #if isempty(flnames)
    gf = [:($fl = Cyanotype.func(cya.$f)) for (f, fl) in zip(fnames, flnames)]
    #kwf = Expr(:parameters, gf...)
    :(getfluxfields(cya::$(sname)) = (; $(gf...),))
end
=#
