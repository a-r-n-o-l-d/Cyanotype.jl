@cyanotype constructor=false begin
    """

    """
    struct MbConvBp <: AbstractConvBp
        skip
        #dropout
        expn
        dwise
        excitation
        projection
    end
end

function MbConvBp(; stride, ch_expn, se_reduction, skip=stride == 1, act=relu,
                  norm=BatchNormBp(act=act), kwargs...)

    stride ∈ [1, 2] || error("`stride` has to be 1 or 2 for `MbConvBp`")

    expn = ChannelExpansionConvBp(; act=act,
    expn=ch_expn,
                                         norm=norm,
                                         kwargs...)

    dwise = DepthwiseConvBp(; act=act,
                                  stride=stride,
                                  norm=norm,
                                  kwargs...)

    excitation = SqueezeExcitationBp(; act=act,
                                       gate_act=hardσ,
                                       reduction=se_reduction * ch_expn,
                                       kwargs...)

    projection = PointwiseConvBp(; norm=norm, kwargs...)

    MbConvBp(skip, expn, dwise, excitation, projection)
end

#function make(bp::MbConvBp, ksize, channels, dropout=0.0)
function make(bp::MbConvBp, ksize, channels, dropout=0) # add dropout for stochastic depth
    in_chs, out_chs = channels
    mid_chs = in_chs * bp.expn.expn
    #=if bp.skip && in_chs !== out_chs
        error("""
        If a 'MbConvBp' have a skip connection defined, the number fo input channels and
        output channels must be the same.
        """)
    end=#
    layers = flatten_layers(
        [
            make(bp.expn, in_chs),
            make(bp.dwise, ksize, mid_chs),
            make(bp.excitation, mid_chs),
            make(bp.projection, mid_chs => out_chs)
        ]
    )
    if bp.skip && in_chs == out_chs
        if iszero(dropout)
            SkipConnection(Chain(layers...), +)
        else
            d = bp.projection.conv.vol ? 5 : 4
            Parallel(+, Chain(layers...), Dropout(dropout, dims=d))
        end
    else
        layers
    end
    #bp.skip && in_chs == out_chs ? SkipConnection(Chain(layers...), +) : layers
end

function make(bp::MbConvBp, ksize, channels::Int, dropout=0)
    make(bp, ksize, channels => channels, dropout)
end

function _mblayers(bp::MbConvBp, ksize, channels)
    in_chs, out_chs = channels
    mid_chs = in_chs * bp.expn.expn
    flatten_layers(
        [
            make(bp.expn, in_chs),
            make(bp.dwise, ksize, mid_chs),
            make(bp.excitation, mid_chs),
            make(bp.projection, mid_chs => out_chs)
        ]
    )
end
