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
    normalization::N = CyNoNorm() #mettre activation dans CyNoNorm? ou mettre activation dans build de batchnorm etc...
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
    @volumetric #enlever
    convolution1::C1
    convolution2::C2 = convolution1
end
)

# channels::Pair in_chs=>out_chs out_chs=>out_chs
# channels::NTuple{3} in_chs=>mid_chs mid_chs=>out_chs
function build(cy::CyDoubleConv, ksize, channels)
    # convolution1.volumetric == convolution2.volumetric || error("")
    c1 = cyanotype(cy.convolution1; volumetric = cy.volumetric)
    c2 = cyanotype(cy.convolution2; volumetric = cy.volumetric)
    in_chs, mid_chs, out_chs = channels
    [build(c1, ksize, in_chs=>mid_chs)..., build(c2, ksize, mid_chs=>out_chs)...]
end

build(cy::CyDoubleConv; ksize, channels) = build(cy, ksize, channels)

# Peut-etre inutile
@cyanotype (
"""
Template describing a module with N `CyConv` repeated.
"""
) (
struct CyNConv{C<:AbstractCyConv} <: AbstractCyConv
    #@activation(relu)
    #@volumetric
    convolution::C # = CyConv() # = ntuple(i -> CyConv(), N)
    nrepeat::Int
    #order::NTuple{N,Int}
end
)

#CyNConv(convolution, nrepeat) = CyNConv{nrepeat}(convolution, nrepeat)

function build(cy::CyNConv, ksize, channels)
    layers = []
    in_chs, out_chs = channels
    for _ in 1:cy.nrepeat
        push!(layers, build(cy.convolution, ksize, in_chs=>out_chs)...)
        in_chs = out_chs
    end
    layers
end


#=function nconv(n, conv = CyConv())
    ntuple(i -> conv, n)
end

function CyNConv(; activation, volumetric, normalization, reverse_norm, pre_activation, use_bias, init, pad, dilation, groups)

end=#

@cyanotype (
"""
aka Hybrid Dilated Convolution
[paper](@ref https://doi.org/10.1109/WACV.2018.00163)
[example](@ref https://doi.org/10.1016/j.image.2019.115664)
[example](@ref https://doi-org/10.1109/ICMA54519.2022.9855903)
"""
) (
struct CyHybridAtrouConv{N,C<:CyConv} <: AbstractCyConv
    dilation_rates::NTuple{N,Int} = (1, 2, 3)
    convolution::C = CyConv(; normalization = CyBatchNorm())
end
)

function build(cy::CyHybridAtrouConv, ksize, channels)
    _check_dilation_rates(ksize, cy.dilation_rates) || error("Invalid dilation rates.")
    layers = []
    in_chs, out_chs = channels
    for dr in cy.dilation_rates
        c = cyanotype(cy.convolution; dilation = dr)
        push!(layers, build(c, ksize, in_chs=>out_chs)...)
        in_chs = out_chs
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

#_check_dilation_rates(3, [1, 2, 3])
#_check_dilation_rates(3, [1, 2, 9]) || println("pouet")
#_check_dilation_rates(3, [3, 2, 1])
