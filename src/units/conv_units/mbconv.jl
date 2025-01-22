@cyanotype constructor=false begin
    """

    """
    struct MbConvBp <: AbstractConvBp
        skip
        expn
        conv
        excn
        proj
    end
end

function MbConvBp(; stride, ch_expn, se_reduction, skip=stride == 1, act=relu,
                  norm=BatchNormBp(act=act), kwargs...)
    stride ∈ [1, 2] || error("`stride` has to be 1 or 2 for `MbConvBp`")
    expn = ChannelExpansionConvBp(; act=act, expn=ch_expn, norm=norm, kwargs...)
    conv = DepthwiseConvBp(; act=act, stride=stride, norm=norm, kwargs...)
    excn = SqueezeExcitationBp(; act=act, gate_act=hardσ, reduc=se_reduction * ch_expn,
                               kwargs...)
    proj = PointwiseConvBp(; norm=norm, kwargs...)
    MbConvBp(skip, expn, conv, excn, proj)
end

function make(bp::MbConvBp, ksize, channels; dropout=0) # add dropout for stochastic depth
    in_chs, out_chs = channels
    mid_chs = in_chs * bp.expn.expn
    layers = flatten_layers(
        [
            make(bp.expn, in_chs),
            make(bp.conv, ksize, mid_chs),
            make(bp.excn, mid_chs),
            make(bp.proj, mid_chs => out_chs)
        ]
    )
    if bp.skip && in_chs == out_chs
        if !iszero(dropout)
            #return SkipConnection(Chain(layers...), +)
        #else
            d = bp.proj.conv.vol ? 5 : 4
            push!(layers, Dropout(dropout, dims=d))
            #return SkipConnection(Chain([layers, Dropout(dropout, dims=d)]...), +)
            #return Parallel(+, Chain(layers...), Dropout(dropout, dims=d))
        end
        return SkipConnection(Chain(layers...), +)
    else
        return layers
    end
end

make(bp::MbConvBp, ksize, channels::Int; dropout=0) = make(
    bp, ksize, channels => channels, dropout=dropout
)
