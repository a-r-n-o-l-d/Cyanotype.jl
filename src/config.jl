using Configurations: codegen_option_type

abstract type AbstractCyano end

mapping(cfg::AbstractCyano) = error("pouet 404")

build(cfg::AbstractCyano) = error("pouet 404")

struct KwargsMapping{N, T1 <: NTuple{N, Symbol}, T2 <: NTuple{N, Any}}
    # flux_function
    field_names::T1
    flux_names::T1
    field_types::T1
    field_defaults::T2
end

KwargsMapping(; field_names, flux_names, field_types, field_defaults) = begin
    KwargsMapping(field_names, flux_names, field_types, field_defaults)
end

field_names(km::KwargsMapping) = km.field_names

flux_names(km::KwargsMapping) = km.flux_names

field_types(km::KwargsMapping) = km.field_types

field_defaults(km::KwargsMapping) = km.field_defaults

each_kwargs(km::KwargsMapping) = zip(
    field_names(km),
    flux_names(km),
    field_types(km),
    field_defaults(km)
)

const mappings = Dict{Symbol, KwargsMapping}() #__MAPPINGS__

function register_mapping!(map)
    haskey(mappings, first(map)) && error("Map $(first(map)) already exists in mappings.")
    push!(mappings, map)
end

const empty_map = KwargsMapping((), (), (), ()) # __EMPTY_MAP__

register_mapping!(:empty_map=>empty_map)

macro config(kmap, expr)
    esc(codegen_config(__module__, expr, kmap))
end

macro config(alias::String, kmap, expr)
    esc(codegen_config(__module__, expr, kmap, alias))
end

macro config(expr)
    esc(codegen_config(__module__, expr))
end

macro config(alias::String, expr)
    esc(codegen_config(__module__, expr, :empty_map, alias))
end

function codegen_config(mod, expr, kmap::Symbol = :empty_map, type_alias = nothing)
    expr = macroexpand(mod, expr)
    def = JLKwStruct(expr, type_alias)
    cname = def.name
    fields = def.fields
    # Generate fields from kmap
    for (name, _, T, default) ∈ each_kwargs(mappings[kmap])
        f = JLKwField(;name = name, type = T, default = default)
        push!(fields, f)
    end
    # If alias is defined add field config
    if !isnothing(type_alias)
        f = JLKwField(;name = :config, type = Reflect)
        push!(fields, f)
    end
    # Generate accessor (getter) functions
    accessors = []
    for f in fields
        d = "Access to the field '$(f.name)' of a '$cname' object."
        a = JLFunction(;name = f.name, args = [:(cfg::$(cname))], body = :(cfg.$(f.name)), doc = d)
        push!(accessors, codegen_ast(a))
    end
    mapfunc = :(mapping(cfg::$(cname)) = $(mappings[kmap]))
    # Copy constructor cname(cfg::$cname; kwargs...)
    copycons = quote
        function $cname(cfg::$cname; kwargs...)
            new_cfg = to_dict(cfg)
            for k ∈ keys(kwargs)
                new_cfg[string(k)] = kwargs[k]
            end
            from_dict(typeof(cfg), new_cfg)
        end
    end
    Expr(:block, codegen_option_type(mod, def), copycons, mapfunc, accessors...)
end

function currate_kwargs(cfg::AbstractCyano, kmap = mapping(cfg))
    #kmap = mapping(cfg)
    kwargs = str2sym(to_dict(cfg))
    # Remove args that are not for Flux
    filter!(k -> first(k) ∈ field_names(kmap), kwargs)
    # Replace field_name by flux_name
    for (field_name, flux_name) ∈ each_kwargs(kmap)
        if haskey(kwargs, field_name) && field_name != flux_name
            tmp = kwargs[field_name]
            delete!(kwargs, field_name)
            kwargs[flux_name] = tmp
        end
    end
    kwargs
end
