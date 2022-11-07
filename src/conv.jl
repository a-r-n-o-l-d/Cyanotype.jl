abstract type AbstractConvBp <: AbstractBlueprint end

const CyPad = Union{SamePad,Int}

@cyanotype (
"""
    ConvBp(; kwargs)

A cyanotype blueprint describing a convolutionnal module or layer depending om the value of
`normalization` argument.
"""
) (
KwargsMapping(;
    flux_function  = :Conv,
    field_names    = (:init,               :pad,      :dilation, :groups),
    flux_kwargs    = (:init,               :pad,      :dilation, :groups),
    field_types    = (:I,                  :P,        Int,      Int),
    def_values     = (Flux.glorot_uniform, Flux.SamePad(), 1,         1)
    )
) (
struct ConvBp{N<:AbstractNormBp,I<:Function,P<:CyPad} <: AbstractConvBp
    @volumetric
    """
    `normalization`:
    """
    normalization::N = NoNormBp()
    """
    `reverse_norm`:
    """
    reverse_norm::Bool = false
    """
    `pre_activation`:
    """
    pre_activation::Bool = false
    """
    `use_bias`:
    """
    use_bias::Bool = normalization isa NoNormBp
end
)

function make(bp::ConvBp; ksize, channels)
    k = bp.volumetric ? (ksize, ksize, ksize) : (ksize, ksize)
    _build_conv(bp.normalization, bp, k, channels) #|> flatten_layers
end

@cyanotype (
"""
    DoubleConvBp(; kwargs)
Describes a convolutionnal module formed by two successive convolutionnal modules.
"""
) (
struct DoubleConvBp{C1<:AbstractConvBp,C2<:AbstractConvBp} <: AbstractConvBp
    @volumetric #enlever
    convolution1::C1
    convolution2::C2 = convolution1
end
)

# channels::Pair in_chs=>out_chs out_chs=>out_chs
# channels::NTuple{3} in_chs=>mid_chs mid_chs=>out_chs
function make(bp::DoubleConvBp; ksize, channels)
    # convolution1.volumetric == convolution2.volumetric || error("")
    c1 = cyanotype(bp.convolution1; volumetric = bp.volumetric)
    c2 = cyanotype(bp.convolution2; volumetric = bp.volumetric)
    in_chs, mid_chs, out_chs = channels
    [
        make(c1; ksize = ksize, channels = in_chs=>mid_chs)...,
        make(c2; ksize = ksize, channels = mid_chs=>out_chs)...
    ]
end

# Peut-etre inutile
@cyanotype (
"""
Template describing a module with N `NConvBp` repeated.
"""
) (
struct NConvBp{C<:AbstractConvBp} <: AbstractConvBp
    convolution::C
    nrepeat::Int
end
)

function make(bp::NConvBp, ksize, channels)
    layers = []
    in_chs, out_chs = channels
    for _ in 1:bp.nrepeat
        push!(layers, make(bp.convolution; ksize = ksize, channels = in_chs=>out_chs)...)
        in_chs = out_chs
    end
    layers
end

@cyanotype (
"""
aka Hybrid Dilated Convolution
[paper](@ref https://doi.org/10.1109/WACV.2018.00163)
[example](@ref https://doi.org/10.1016/j.image.2019.115664)
[example](@ref https://doi-org/10.1109/ICMA54519.2022.9855903)
"""
) (
struct HybridAtrouConvBp{N,C<:ConvBp} <: AbstractConvBp
    dilation_rates::NTuple{N,Int} = (1, 2, 3)
    convolution::C = ConvBp(; normalization = BatchNormBp())
end
)

function make(bp::HybridAtrouConvBp, ksize, channels)
    _check_dilation_rates(ksize, bp.dilation_rates) || error("Invalid dilation rates.")
    layers = []
    in_chs, out_chs = channels
    for dr in bp.dilation_rates
        c = cyanotype(bp.convolution; dilation = dr)
        push!(layers, make(c; ksize = ksize, channels = in_chs=>out_chs)...)
        in_chs = out_chs
    end
    layers
end


#_check_dilation_rates(3, [1, 2, 3])
#_check_dilation_rates(3, [1, 2, 9]) || println("pouet")
#_check_dilation_rates(3, [3, 2, 1])

############################################################################################
#                                   INTERNAL FUNCTIONS                                     #
############################################################################################

# A usual convolutionnal layer
function _build_conv(::NoNormBp, bp, k, chs)
    [Conv(k, chs, bp.normalization.activation; kwargs(bp)...)]
end

# Convolutionnal module: convolutionnal layer & normalization layer
function _build_conv(nm, bp, k, chs)
    layers = []
    in_chs, out_chs = chs
    activation = bp.normalization.activation
    # Normalization first
    if bp.reverse_norm
        # Activation before convolution ?
        if bp.pre_activation
            act_n = activation
            act_c = identity
        else
            act_n = identity
            act_c = activation
        end
        norm = cyanotype(nm; activation = act_n)
        conv = Conv(k, chs, act_c; bias = bp.use_bias, kwargs(bp)...)
        push!(layers, make(norm; channels = in_chs), conv)
    # Convolution first
    else
        # Activation before convolution ?
        if bp.pre_activation
            act_n = identity
            push!(layers, activation)
        else
            act_n = activation
        end
        norm = cyanotype(nm; activation = act_n)
        conv = Conv(k, chs; bias = bp.use_bias, kwargs(bp)...)
        push!(layers, conv, make(norm; channels =  out_chs))
    end
    layers
end

#https://arxiv.org/pdf/1702.08502.pdf
# DOI 10.1109/WACV.2018.00163
function _check_dilation_rates(k, dr)
    issorted(dr) || error("Dilation rates must be increasing.")
    M = dr[end]
    i = length(dr)
    while i > 2
        i = i - 1
        M = max(M - 2 * dr[i], M - 2 * (M - dr[i]), dr[i])
    end
    M <= k
end
