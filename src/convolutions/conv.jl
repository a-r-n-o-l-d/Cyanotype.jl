abstract type AbstractBpConv <: AbstractBlueprint end

const CyPad = Union{SamePad,Int}

@cyanotype begin
    KwargsMapping(
            flfunc = :Conv,
            fnames = (:stride, :pad,      :dilation, :groups, :init), #stride, bias
            flargs = (:stride, :pad,      :dilation, :groups, :init),
            ftypes = (Int,     :P,        Int,       Int,     :I),
            defval = (1,       SamePad(), 1,         1,       glorot_uniform)
        )

    """
    BpConv(; kwargs...)

    A cyanotype blueprint describing a convolutionnal module or layer depending om the value
    of `normalization` argument.
    """
    struct BpConv{N<:Union{Nothing,AbstractBpNorm},A,I<:Function,P<:CyPad} <: AbstractBpConv
        @volume
        @activation(identity)
        """
        `normalization`:
        """
        normalization::N = nothing
        """
        `depthwise`:
        """
        depthwise::Bool = false
        # residual
        """
        `revnorm`:
        """
        revnorm::Bool = false
        """
        `preactivation`:
        """
        preactivation::Bool = false
        """
        `bias`:
        """
        bias::Bool = normalization isa Nothing
    end
end

make(bp::BpConv, ksize, channels::Int) = make(bp, ksize, channels => channels)

# Regular convolutionnal layer
function make(bp::BpConv{<:Nothing}, ksize, channels::Pair)
    k = genk(ksize, bp.volume)
    kw = kwargs(bp)
    if bp.depthwise
        kw[:groups] = first(channels)
    end
    Conv(k, channels, bp.activation; kw...) #|> flatten_layers
end

# Convolutionnal unit: convolutionnal layer & normalization layer
function make(bp::BpConv{<:AbstractBpNorm}, ksize, channels::Pair)
    k = genk(ksize, bp.volume)
    layers = []
    in_chs, out_chs = channels
    activation = bp.normalization.activation
    kw = kwargs(bp)
    if bp.depthwise
        kw[:groups] = in_chs
    end
    # Normalization first
    if bp.revnorm
        # Activation before convolution ?
        if bp.preactivation
            act_n = activation
            act_c = identity
        else
            act_n = identity
            act_c = activation
        end
        norm = cyanotype(bp.normalization; activation = act_n)
        conv = Conv(k, channels, act_c; bias = bp.bias, kw...)
        push!(layers, make(norm, in_chs), conv)
    # Convolution first
    else
        # Activation before convolution ?
        if bp.preactivation
            act_n = identity
            push!(layers, activation)
        else
            act_n = activation
        end
        norm = cyanotype(bp.normalization; activation = act_n)
        conv = Conv(k, channels; bias = bp.bias, kw...)
        push!(layers, conv, make(norm, out_chs))
    end
    flatten_layers(layers)
end
