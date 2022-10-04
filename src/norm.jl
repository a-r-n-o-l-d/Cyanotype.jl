abstract type AbstractCyNorm <: AbstractCyanotype end

# Defines kwargs for Flux normalisation layers
const _NORMKW = (
    field_names = (:init_shift,  :init_scale, :affine, :track_stats, :epsilon, :momentum), #init_bias => init_shift
    flux_kwargs = (:initβ,       :initγ,      :affine, :track_stats, :ϵ,       :momentum),
    field_types = (:I1,          :I2,         Bool,    Bool,         :F,       :F), # Function => type instbility, wrap in CyFunc => curate doit etre modifee
    def_values  = (Flux.zeros32, Flux.ones32, true,    true,         1f-5,     0.1f0)
)

#="""
    CyanoIdentityNorm()

Tagging `struct` indicating that no normalisation layer should be used in a building process.
"""
CyIdentityNorm=#

@cyanotype """
Tagging `struct` indicating that no normalisation layer should be used in a building process.
""" struct CyIdentityNorm <: AbstractCyNorm end

"""
$(autogen_build_doc(CyIdentityNorm, false, true))
"""
build(::Any, ::CyIdentityNorm) = Flux.identity #inutile


register_mapping!(:bnmap=>KwargsMapping(; flux_function = :BatchNorm, _NORMKW...))

@cyanotype bnmap """
Wraps a Flux.Batchnorm
""" struct CyBatchNorm{F<:CyFloat,A<:Function,I1<:Function,I2<:Function} <: AbstractCyNorm
    @activation(relu)
end

# auto generation doc for build
function build(channels, cy::CyBatchNorm)
    #kwargs = curate(cya)
    BatchNorm(channels, cy.activation; kwargs(cy)...)
end

register_mapping!(:gnmap=>KwargsMapping(; flux_function = :GroupNorm, _NORMKW...))

@cyanotype gnmap """

""" struct CyGroupNorm{F<:CyFloat,A<:Function,I1<:Function,I2<:Function} <: AbstractCyNorm
    @activation(relu)
    """
    `groups`: the number of groups passed to [`GroupNorm`](@ref Flux.GroupNorm) constructor
    """
    groups::Int
end

function build(channels, cy::CyGroupNorm)
    #kwargs = curate(cy)
    GroupNorm(channels, cy.groups, cy.activation; kwargs(cy)...)
end
