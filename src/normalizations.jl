abstract type AbstractBpNorm <: AbstractBlueprint end
#=
@cyanotype begin
    """
    NoNormalization()
    """
    struct BpNoNorm{A} <: AbstractBpNorm
        @activation(Flux.relu)
    end
end
=#
@cyanotype begin
    KwargsMapping(
            flfunc = :BatchNorm,
            fnames = (:initshift, :initscale, :affine, :trackstats,  :epsilon, :momentum),
            flargs = (:initβ,     :initγ,     :affine, :track_stats, :ϵ,       :momentum),
            ftypes = (:I1,        :I2,        Bool,    Bool,         :F,       :F),
            defval  = (zeros32,   ones32,     true,    true,         1f-5,     0.1f0)
        )

    """
    Wraps a Flux.Batchnorm
    """
    struct BpBatchNorm{F<:CyFloat,A<:Function,I1<:Function,
                              I2<:Function} <: AbstractBpNorm
        @activation(relu)
    end
end

function make(bp::BpBatchNorm; channels)
    [BatchNorm(channels, bp.activation; kwargs(bp)...)]
end

@cyanotype begin
    # !!!! `track_stats=true` will be removed from GroupNorm in Flux 0.14.
    KwargsMapping(
            flfunc = :GroupNorm,
            fnames = (:initshift, :initscale, :affine, :trackstats,  :epsilon, :momentum),
            flargs = (:initβ,     :initγ,     :affine, :track_stats, :ϵ,       :momentum),
            ftypes = (:I1,        :I2,         Bool,    Bool,        :F,       :F),
            defval  = (zeros32,   ones32,      true,    false,       1f-5,     0.1f0)
        )

    """
        BpGroupNorm(; kwargs...)

    Describes a building process for a [`Groupnorm`](@ref Flux.Groupnorm) layer.
    make(channels, bp::CyGroupNorm)
    """
    struct BpGroupNorm{F<:CyFloat,A<:Function,I1<:Function,I2<:Function} <: AbstractBpNorm
        @activation(relu)
        """
        `groups`: the number of groups passed to [`GroupNorm`](@ref Flux.GroupNorm)
        constructor
        """
        groups::Int
    end
end

function make(bp::BpGroupNorm; channels)
    [GroupNorm(channels, bp.groups, bp.activation; kwargs(bp)...)]
end

@cyanotype begin
    KwargsMapping(
            flfunc = :InstanceNorm,
            fnames = (:initshift, :initscale, :affine, :trackstats,  :epsilon, :momentum),
            flargs = (:initβ,     :initγ,     :affine, :track_stats, :ϵ,       :momentum),
            ftypes = (:I1,        :I2,        Bool,    Bool,         :F,       :F),
            defval  = (zeros32,    ones32,    false,   false,        1f-5,     0.1f0)
        )

    """
        BpInstanceNorm(; kwargs...)

    Describes a building process for a [`InstanceNorm`](@ref) layer.
    make(channels, bp::CyInstanceNorm)
    """
    struct BpInstanceNorm{F<:CyFloat,A<:Function,I1<:Function,
                                 I2<:Function} <: AbstractBpNorm
        @activation(relu)
    end
end

function make(bp::BpInstanceNorm; channels)
    [InstanceNorm(channels, bp.activation; kwargs(bp)...)]
end
