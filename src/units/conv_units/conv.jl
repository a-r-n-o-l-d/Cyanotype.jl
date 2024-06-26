@cyanotype begin
    KwargsMapping(
            flfunc = :Conv,
            fnames = (:stride, :pad,      :dila,     :groups, :init),
            flargs = (:stride, :pad,      :dilation, :groups, :init),
            defval = (1,       SamePad(), 1,         1,       glorot_uniform)
    )

    """
    ConvBp(; kwargs...)

    A cyanotype blueprint describing a convolutionnal module or layer depending om the value
    of `norm` argument.
    """
    struct ConvBp{N<:Union{Nothing,AbstractNormBp}} <: AbstractConvBp
        @volume
        @activation(identity)
        """
        `norm`:
        """
        norm::N = nothing
        """
        `dwise`:
        """
        dwise = false
        """
        `rvnorm`:
        """
        rvnorm = false
        """
        `preact`:
        """
        preact = false
        """
        `bias`:
        """
        bias = norm isa Nothing
    end
end

make(bp::ConvBp, ksize, channels::Int) = make(bp, ksize, channels => channels)

# Regular convolutionnal layer
function make(bp::ConvBp{N}, ksize, channels::Pair) where N<:Nothing
    k = genk(ksize, bp.vol)
    kw = kwargs(bp)
    if bp.dwise
        kw[:groups] = first(channels)
    end
    Conv(k, channels, bp.act; kw...)
end

# Convolutionnal unit: convolutionnal layer & norm layer
function make(bp::ConvBp{N}, ksize, channels::Pair) where N<:AbstractNormBp
    k = genk(ksize, bp.vol)
    layers = []
    in_chs, out_chs = channels
    act = bp.act
    kw = kwargs(bp)
    if bp.dwise
        kw[:groups] = in_chs
    end
    # Normalization first
    if bp.rvnorm
        # Activation before convolution ?
        if bp.preact
            act_n = act
            act_c = identity
        else
            act_n = identity
            act_c = act
        end
        norm = cyanotype(bp.norm; act = act_n)
        conv = Conv(k, channels, act_c; bias = bp.bias, kw...)
        push!(layers, make(norm, in_chs), conv)
    # Convolution first
    else
        # Activation before convolution ?
        if bp.preact
            act_n = identity
            push!(layers, act)
        else
            act_n = act
        end
        norm = cyanotype(bp.norm; act = act_n)
        conv = Conv(k, channels; bias = bp.bias, kw...)
        push!(layers, conv, make(norm, out_chs))
    end
    flatten_layers(layers)
end
