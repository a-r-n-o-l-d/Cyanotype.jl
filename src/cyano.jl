abstract type AbstractCyano end

struct KwargsMapping{N,T1<:NTuple{N,Symbol},T2<:NTuple{N,Any}}
    flux_function::Symbol
    field_names::T1
    flux_names::T1
    field_types::T1
    field_defaults::T2
    flux_doc::MD
    additional_doc::String
end

function KwargsMapping(; flux_function = :notflux, field_names = (), flux_names = (),
                         field_types = (), field_defaults = (), additional_doc = "")
    # Generate flux documentation if necessary
    if flux_function == :notflux
        flux_doc = MD()
    else
        flux_doc = eval(macroexpand(Flux, :(@doc $flux_function)))
    end
    KwargsMapping(flux_function, field_names, flux_names, field_types, field_defaults,
                  flux_doc, additional_doc)
end

eachkwargs(km::KwargsMapping) = zip(km.field_names, km.flux_names, km.field_types,
                                    km.field_defaults)

const MAPPINGS::Dict{Symbol,KwargsMapping} = Dict(:empty_map => KwargsMapping())

function register_mapping!(mapping)
    haskey(MAPPINGS, first(mapping)) && error(
        "Map $(first(mapping)) already exists in MAPPINGS."
    )
    push!(MAPPINGS, mapping)
end

function curate(cfg::AbstractCyano, mod = @__MODULE__)
    kmap = mod.mapping(cfg)
    fields = mod.getfields(cfg)
    kwargs = Dict(pairs(fields))
    # Remove args that are not for Flux
    filter!(k -> first(k) ∈ kmap.field_names, kwargs)
    # Replace field_name by flux_name
    for (field_name, flux_name) ∈ eachkwargs(kmap)
        if haskey(kwargs, field_name) && field_name != flux_name
            tmp = kwargs[field_name]
            delete!(kwargs, field_name)
            kwargs[flux_name] = tmp
        end
    end
    kwargs
end

macro cyano(expr)
    expr = macroexpand(__module__, expr)
    esc(_cyano_struct(:empty_map, expr.args[2], expr.args[3]))
end

macro cyano(kmap, expr)
    expr = macroexpand(__module__, expr)
    esc(_cyano_struct(kmap, expr.args[2], expr.args[3]))
end

############################################################################################
#                                  INTERNAL FUNCTIONS                                      #
############################################################################################

function _cyano_struct(kmap, head, body)
    # Forces the struct to inherit from AbstractCyano
    if head isa Symbol || head.head === :curly
        # It is not type stable to do that, since the head type is changed, but at this
        # stage performance is not a big deal.
        head = Expr(:<:, head, AbstractCyano)
    end

    # Flux name
    flname = Cyanotype.MAPPINGS[kmap].flux_function

    # Flux doc
    fldoc = Cyanotype.MAPPINGS[kmap].flux_doc

    # Additional doc
    adddoc = Cyanotype.MAPPINGS[kmap].additional_doc

    # Flux ref for documentation
    flref = "[`$flname`](@ref Flux.$flname)"

    # Cyanotype name
    cyaname = _struct_name(head)

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
        elseif f isa Expr && f.head === :string
            #println(f.args)
            tmpdoc = eval(f.args[1]) # beurk
        else
            _push_field!(fields, kwargs, fnames, fdocs, _parse_body_field(f, tmpdoc)...)
            tmpdoc = ""
        end
    end

    # Adds fields defined by kmap
    for (name, flname, T, def) in eachkwargs(Cyanotype.MAPPINGS[kmap])
        doc = "`$name` corresponds to `$flname` in $flref with a default value `$def`"
        _push_field!(fields, kwargs, fnames, fdocs, name, T, def, doc)
    end

    # Generates the struct and its associated functions
    if isempty(fdocs) && isempty(adddoc)
        quote
            struct $head
                $(fields...)
            end
            $(_kwargs_constructor(cyaname, fnames, kwargs))
            $(_copy_constructor(cyaname))
            $(_mapping_func1(cyaname, kmap))
            $(_mapping_func2(cyaname, kmap))
            $(_getfields_func(cyaname, fnames))
        end
    else
        quote
            """$($(_generate_documentation(cyaname, fdocs, flname, flref, fldoc, adddoc)))"""
            struct $head
                $(fields...)
            end
            $(_kwargs_constructor(cyaname, fnames, kwargs))
            $(_copy_constructor(cyaname))
            $(_mapping_func1(cyaname, kmap))
            $(_mapping_func2(cyaname, kmap))
            $(_getfields_func(cyaname, fnames))
        end
    end
end

# All of the functions below are just helper functions, which break the logic of
# _cyano_struct into small pieces to avoid a single large function. Perhaps they could be
# inlined.

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
            error("Blocks are not allowed in body definition.")
        end
    end
    if isempty(doc)
        doc = "`$name` is not documented"
    end
    name, T, def, doc
end

function _push_field!(fields, kwargs, fnames, fdocs, name, T, def, doc)
    push!(fields, :($name::$T))
    if isnothing(def)
        push!(kwargs, :($name))
    else
        push!(kwargs, Expr(:kw, name, :($name=$def)))
    end
    push!(fnames, name)
    push!(fdocs, doc)
end

function _generate_documentation(cyaname, fdocs, flname, flref, fldoc, adddoc)
    #fldoc = ""
    if flname == :notflux
        docs = adddoc
    else
        docs =
        """
            $cyaname(; kwargs...)
        Describes a building process for $flref. $adddoc

        See also [`build`](@ref)

        Keyword arguments:\n
        """
    end
    for d in fdocs
        docs = "$docs - $d\n"
    end
    # If this a wrapper for a flux function/constructor, the flux documentation is added.
    # This is helpfull (or not?), to avoid to switch from one doc to another.
    if !isempty(fldoc)
        docs = "$docs Flux documentation:\n $fldoc"
    end
    docs
end

function _kwargs_constructor(sname, fnames, kwargs)
    # If there is no field in the struct, no copy constructor
    if isempty(fnames) && isempty(kwargs)
        :()
    else
        kw = Expr(:parameters, kwargs...)
        :($sname($kw) = $sname($(fnames...)))
    end
end

function _copy_constructor(sname)
    quote
        function $sname(cya::$sname; kwargs...)
            args = []
            fields = getfields(cya)
            for k in keys(fields)
                if haskey(kwargs, k)
                    push!(args, kwargs[k])
                else
                    push!(args, fields[k])
                end
            end
            typeof(cya)(args...)
        end
    end
end

_mapping_func1(sname, kmap) = :(mapping(::$sname) = $(Cyanotype.MAPPINGS[kmap]))

_mapping_func2(sname, kmap) = :(mapping(::Type{$sname}) = $(Cyanotype.MAPPINGS[kmap]))

function _getfields_func(sname, fnames)
    gf = [:($f = cya.$f) for f in fnames]
    #kwf = Expr(:parameters, gf...)
    :(getfields(cya::$(sname)) = (; $(gf...),))
end
