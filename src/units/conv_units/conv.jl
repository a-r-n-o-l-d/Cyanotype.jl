

const CyPad = Union{SamePad,Int}

@cyanotype begin
    KwargsMapping(
            flfunc = :Conv,
            fnames = (:stride, :pad,      :dilation, :groups, :init), #stride, bias
            flargs = (:stride, :pad,      :dilation, :groups, :init),
            ftypes = (Int,     Union{SamePad,Int},        Int,       Int,     Any),
            defval = (1,       SamePad(), 1,         1,       glorot_uniform)
    )

    """
    ConvBp(; kwargs...)

    A cyanotype blueprint describing a convolutionnal module or layer depending om the value
    of `normalization` argument.
    """
    struct ConvBp{N<:Union{Nothing,AbstractNormBp}#=,A,I<:Function,P<:Union{SamePad,Int}=#} <: AbstractConvBp #<:Function
        volume = false
        activation = identity
        """
        `normalization`:
        """
        normalization::N = nothing
        """
        `depthwise`:
        """
        depthwise#=::Bool=# = false
        """
        `revnorm`:
        """
        revnorm#=::Bool=# = false
        """
        `preactivation`:
        """
        preactivation#=::Bool=# = false
        """
        `bias`:
        """
        bias#=::Bool=# = normalization isa Nothing
    end
end

make(bp::ConvBp, ksize, channels::Int) = make(bp, ksize, channels => channels)

# Regular convolutionnal layer
function make(bp::ConvBp{<:Nothing}, ksize, channels::Pair)
    k = genk(ksize, bp.volume)
    kw = kwargs(bp)
    if bp.depthwise
        kw[:groups] = first(channels)
    end
    Conv(k, channels, bp.activation; kw...) #|> flatten_layers
end

# Convolutionnal unit: convolutionnal layer & normalization layer
function make(bp::ConvBp{<:AbstractNormBp}, ksize, channels::Pair)
    k = genk(ksize, bp.volume)
    layers = []
    in_chs, out_chs = channels
    activation = bp.activation
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
