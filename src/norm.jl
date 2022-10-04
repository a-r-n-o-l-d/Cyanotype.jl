abstract type AbstractCyNorm <: AbstractCyanotype end

# Defines kwargs for Flux normalisation layers
#=const _NORMKW = (
    field_names = (:init_shift,  :init_scale, :affine, :track_stats, :epsilon, :momentum), #init_bias => init_shift
    flux_kwargs = (:initβ,       :initγ,      :affine, :track_stats, :ϵ,       :momentum),
    field_types = (:I1,          :I2,         Bool,    Bool,         :F,       :F), # Function => type instbility, wrap in CyFunc => curate doit etre modifee
    def_values  = (Flux.zeros32, Flux.ones32, true,    true,         1f-5,     0.1f0)
)=#

#="""
    CyanoIdentityNorm()

Tagging `struct` indicating that no normalisation layer should be used in a building process.
"""
CyIdentityNorm=#

@cyanotype """
    CyNoNorm()
Tagging `struct` indicating that no normalisation layer should be used in a building
process.
""" struct CyNoNorm <: AbstractCyNorm end

#="""
$(autogen_build_doc(CyIdentityNorm, false, true))
"""
build(::Any, ::CyIdentityNorm) = Flux.identity #inutile=#


register_mapping!(:bnmap=>KwargsMapping(; flux_function = :BatchNorm,
    field_names = (:init_shift,  :init_scale, :affine, :track_stats, :epsilon, :momentum),
    flux_kwargs = (:initβ,       :initγ,      :affine, :track_stats, :ϵ,       :momentum),
    field_types = (:I1,          :I2,         Bool,    Bool,         :F,       :F),
    def_values  = (Flux.zeros32, Flux.ones32, true,    true,         1f-5,     0.1f0)))

@cyanotype bnmap """
Wraps a Flux.Batchnorm
""" struct CyBatchNorm{F<:CyFloat,A<:Function,I1<:Function,I2<:Function} <: AbstractCyNorm
    @activation(Flux.relu)
end

# auto generation doc for build
function build(cy::CyBatchNorm, channels)
    BatchNorm(channels, cy.activation; kwargs(cy)...)
end

build(cy::CyBatchNorm; channels) = build(cy, channels)

# !!!! `track_stats=true` will be removed from GroupNorm in Flux 0.14.
register_mapping!(:gnmap=>KwargsMapping(; flux_function = :GroupNorm,
    field_names = (:init_shift,  :init_scale, :affine, :track_stats, :epsilon, :momentum),
    flux_kwargs = (:initβ,       :initγ,      :affine, :track_stats, :ϵ,       :momentum),
    field_types = (:I1,          :I2,         Bool,    Bool,         :F,       :F),
    def_values  = (Flux.zeros32, Flux.ones32, true,    false,        1f-5,     0.1f0)))

@cyanotype gnmap """
    CyGroupNorm(; kwargs...)
Describes a building process for a [`Groupnorm`](@ref Flux.Groupnorm) layer.
build(channels, cy::CyGroupNorm)
""" struct CyGroupNorm{F<:CyFloat,A<:Function,I1<:Function,I2<:Function} <: AbstractCyNorm
    @activation(relu)
    """
    `groups`: the number of groups passed to [`GroupNorm`](@ref Flux.GroupNorm) constructor
    """
    groups::Int
end

# build(cy::CyGroupNorm; channels)
function build(cy::CyGroupNorm, channels)
    #kwargs = curate(cy)
    GroupNorm(channels, cy.groups, cy.activation; kwargs(cy)...)
end

build(cy::CyGroupNorm; channels) = build(cy, channels)


register_mapping!(:inmap=>KwargsMapping(; flux_function = :GroupNorm,
    field_names = (:init_shift,  :init_scale, :affine, :track_stats, :epsilon, :momentum),
    flux_kwargs = (:initβ,       :initγ,      :affine, :track_stats, :ϵ,       :momentum),
    field_types = (:I1,          :I2,         Bool,    Bool,         :F,       :F),
    def_values  = (Flux.zeros32, Flux.ones32, false,    false,        1f-5,     0.1f0)))

@cyanotype gnmap """
    CyInstanceNorm(; kwargs...)
Describes a building process for a [`InstanceNorm`](@ref Flux.InstanceNorm) layer.
build(channels, cy::CyInstanceNorm)
""" struct CyInstanceNorm{F<:CyFloat,A<:Function,I1<:Function,I2<:Function} <: AbstractCyNorm
    @activation(relu)
end

function build(cy::CyInstanceNorm, channels)
    InstanceNorm(channels, cy.activation; kwargs(cy)...)
end

build(cy::CyInstanceNorm; channels) = build(cy, channels)
