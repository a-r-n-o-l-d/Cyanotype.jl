abstract type AbstractCfg end

mapping(cfg::AbstractCfg) = error("pouet 404")

buid(cfg::AbstractCfg) = error("pouet 404")

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

each_option(km::KwargsMapping) = zip(
    field_names(km),
    flux_names(km),
    field_types(km),
    field_defaults(km)
)

const mappings = Dict{Symbol, OptionMapping}()

function register_mapping!(map)
    haskey(mappings, first(map)) && error("Map $(first(map)) already exists in mappings.")
    push!(mappings, map)
end

const empty_map = OptionMapping((), (), (), ())

register_mapping!(:empty_map=>empty_map)
