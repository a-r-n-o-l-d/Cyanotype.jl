abstract type AbstractCfg end

mapping(cfg::AbstractCfg) = error("pouet 404")

build(cfg::AbstractCfg) = error("pouet 404")

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

const mappings = Dict{Symbol, KwargsMapping}()

function register_mapping!(map)
    haskey(mappings, first(map)) && error("Map $(first(map)) already exists in mappings.")
    push!(mappings, map)
end

const empty_map = KwargsMapping((), (), (), ())

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
    for (name, _, T, default) ∈ each_kwargs(mappings[kmap])
        f = JLKwField(;name = name, type = T, default = default)
        #f.doc = ...generation auto... ou générer doc auto pour les accesseurs
        push!(def.fields, f)
    end
    if !isnothing(type_alias)
        f = JLKwField(;name = :config, type = Reflect)
        push!(def.fields, f)
    end
    accessors = []
    for f in def.fields
        #a = JLFunction(;) # ajout docstring
        # codegen_ast(a)
        push!(accessors, :($(f.name)(cfg::$(def.name)) = cfg.$(f.name)))
    end
    mapfunc = :(mapping(cfg::$(def.name)) = $(mappings[kmap]))
    Expr(:block, 
        Configurations.codegen_option_type(mod, def),
        mapfunc,
        accessors...)
end

function currate_kwargs(cfg::AbstractCfg, kmap = mapping(cfg))
    #km = mapping(cfg)
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
