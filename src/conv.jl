abstract type AbstractCyConv <: AbstractCyanotype end

const CyPad = Union{SamePad,Int}

register_mapping!(:convmap=>KwargsMapping(;
    flux_function  = :Conv,
    field_names    = (:init,               :pad,      :dilation, :groups),
    flux_kwargs    = (:init,               :pad,      :dilation, :groups),
    field_types    = (:I,                  :P,        Int,      Int),
    def_values     = (Flux.glorot_uniform, Flux.SamePad(), 1,         1)
))

@cyanotype convmap (
"""
    CyConv(; kwargs)
Describes a convolutionnal module or layer depending om the value of `normalization`
argument.
"""
) (
struct CyConv{N<:AbstractCyNorm,A<:Function,I<:Function,P<:CyPad} <: AbstractCyConv
    @activation(relu)
    @volumetric
    """
    `normalization`:
    """
    normalization::N = CyNoNorm()
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
    use_bias::Bool = normalization isa CyNoNorm
end
)

function build(cy::CyConv, ksize, channels)
    k = cy.volumetric ? (ksize, ksize, ksize) : (ksize, ksize)
    _build_conv(cy.normalization, cy, k, channels) #|> flatten_layers
end

build(cy::CyConv; ksize, channels) = build(cy, ksize, channels)

# A regular convolutionnal layer
function _build_conv(::CyNoNorm, cy, k, chs)
    [Conv(k, chs, cy.activation; kwargs(cy)...)]
end

# Convolutionnal module: convolutionnal layer & normalization layer
function _build_conv(nm, cy, k, chs)
    layers = []
    in_chs, out_chs = chs
    # Normalization first
    if cy.reverse_norm
        # Activation before convolution ?
        if cy.pre_activation
            act_n = cy.activation
            act_c = identity
        else
            act_n = identity
            act_c = cy.activation
        end
        norm = cyanotype(nm; activation = act_n)
        conv = Conv(k, chs, act_c; bias = cy.use_bias, kwargs(cy)...)
        push!(layers, build(norm, in_chs), conv)
    # Convolution first
    else
        # Activation before convolution ?
        if cy.pre_activation
            act_n = identity
            push!(layers, cy.activation)
        else
            act_n = cy.activation
        end
        norm = cyanotype(nm; activation = act_n)
        conv = Conv(k, chs; bias = cy.use_bias, kwargs(cy)...)
        push!(layers, conv, build(norm, out_chs))
    end
    layers
end


@cyanotype (
"""
    CyDoubleConv(; kwargs)
Describes a convolutionnal module formed by two successive convolutionnal modules.
"""
) (
struct CyDoubleConv{C1<:AbstractCyConv,C2<:AbstractCyConv} <: AbstractCyConv
    @volumetric
    convolution1::C1
    convolution2::C2 = convolution1
end
)

# channels::Pair in_chs=>out_chs out_chs=>out_chs
# channels::NTuple{3} in_chs=>mid_chs mid_chs=>out_chs
function build(cy::CyDoubleConv, ksize, channels)
    c1 = cyanotype(cy.convolution1; volumetric = cy.volumetric)
    c2 = cyanotype(cy.convolution2; volumetric = cy.volumetric)
    in_chs, mid_chs, out_chs = channels
    [build(c1, ksize, in_chs=>mid_chs)..., build(c2, ksize, mid_chs=>out_chs)...]
end

build(cy::CyDoubleConv; ksize, channels) = build(cy, ksize, channels)

@cyanotype (
"""
Template describing a module with N `CyConv` repeated.
"""
) (
struct CyNConv{C<:CyConv} <: AbstractCyConv
    #@activation(relu)
    #@volumetric
    convolution::C # = CyConv() # = ntuple(i -> CyConv(), N)
    nrepeat::Int
end
)

function build(cy::CyNConv, ksize, channels)
    layers = []
    in_chs, out_chs = channels
    for i in 1:cy.nrepeat
        push!(layers, build(cy.convolution, ksize, in_chs=>out_chs)...)
        in_chs = out_chs
    end
    layers
end


function nconv(n, conv = CyConv())
    ntuple(i -> conv, n)
end

#=function CyNConv(; activation, volumetric, normalization, reverse_norm, pre_activation, use_bias, init, pad, dilation, groups)

end=#

@cyanotype (
"""
"""
) (
struct CyHybridAtrouConv{N,D<:NTuple{N,Int},C<:NTuple{N,CyConv}} <: AbstractCyConv
    #@activation(relu)
    @volumetric
    dilation_rates::D = (1, 2, 3)
    nconv = N
    convolutions::C  # = ntuple(i -> CyConv(; normalization = CyBatchNorm(), dilation = i), 3)
end
)



# hybrid dilated convolution : 10.1109/WACV.2018.00163
# https://doi.org/10.1016/j.image.2019.115664
# arg chain ?
function hybrid_atrou_conv(k, chs, activation = relu; norm = BatchNorm, dilation_rates, pad = SamePad(), kwargs...) #hybrid_atrou_conv(k, ch, σ = identity; dilation_rates, norm_layer )
    check_dilation_rates(first(k), dilation_rates) || error("Invalid diation rates.")
    in_chs, out_chs = chs
    layers = []
    for r ∈ dilation_rates
        push!(layers, conv(k, in_chs=>out_chs, activation; dilation = r, norm = norm, pad = pad, kwargs...))
        in_chs = out_chs
    end
    flatten_layers(layers)
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

#_check_dilation_rates(3, [1, 2, 3])
#_check_dilation_rates(3, [1, 2, 9]) || println("pouet")
#_check_dilation_rates(3, [3, 2, 1])
