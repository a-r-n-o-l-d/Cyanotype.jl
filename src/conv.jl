abstract type AbstractCyConv <: AbstractCyanotype end

const CyPad = Union{SamePad,Int}

register_mapping!(:convmap=>KwargsMapping(;
    flux_function  = :Conv,
    field_names    = (:init,               :pad,      :dilation, :groups),
    flux_kwargs    = (:init,               :pad,      :dilation, :groups),
    field_types    = (:I,                  :P,        :Int,      :Int),
    def_values     = (Flux.glorot_uniform, Flux.SamePad(), 1,         1)
))

@cyanotype convmap """

""" struct CyConv{N<:AbstractCyNorm,A<:Function,I<:Function,P<:CyPad} <: AbstractCyConv
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
