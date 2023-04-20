@cyanotype constructor=false begin
    """

    """
    struct BpMbConv <: AbstractConvBp
        skip::Bool
        #dropout
        expansion::BpChannelExpansionConv
        depthwise::BpDepthwiseConv
        excitation::BpSqueezeExcitation
        projection::PointwiseConvBp
    end
end

# BpMbConv(expkwargs, depkwargs, exckwargs, projkwargs; stride, ch_expansion, se_reduction, skip=stride == 1, activation=relu,
# normalization=BatchNormBp(activation=activation))
function BpMbConv(; stride, ch_expansion, se_reduction, skip=stride == 1, activation=relu,
                  normalization=BatchNormBp(activation=activation), kwargs...)

    stride ∈ [1, 2] || error("`stride` has to be 1 or 2 for `BpMbConv`")

    expansion = BpChannelExpansionConv(; activation=activation,
                                         expansion=ch_expansion,
                                         normalization=normalization,
                                         kwargs...)

    depthwise = BpDepthwiseConv(; activation=activation,
                                  stride=stride,
                                  normalization=normalization,
                                  kwargs...)

    excitation = BpSqueezeExcitation(; activation=activation,
                                       gate_activation=hardσ,
                                       reduction=se_reduction,
                                       kwargs...)

    projection = PointwiseConvBp(; normalization=normalization, kwargs...)

    BpMbConv(skip, expansion, depthwise, excitation, projection)
end

function make(bp::BpMbConv, ksize, channels) # add dropout for stochastic depth
    in_chs, out_chs = channels
    mid_chs = in_chs * bp.expansion.expansion
    if bp.skip && in_chs !== out_chs
        error("""
        If a 'BpMbConv' have a skip connection defined, the number fo input channels and
        output channels must be the same.
        """)
    end
    layers = flatten_layers(
        [
            make(bp.expansion, in_chs),
            make(bp.depthwise, ksize, mid_chs),
            make(bp.excitation, mid_chs),
            make(bp.projection, mid_chs => out_chs)
        ]
    )
    bp.skip ? SkipConnection(Chain(layers...), +) : layers
end

function _mblayers(bp::BpMbConv, ksize, channels)
    in_chs, out_chs = channels
    mid_chs = in_chs * bp.expansion.expansion
    if bp.skip && in_chs !== out_chs
        error("""
        If a 'BpMbConv' have a skip connection defined, the number fo input channels and
        output channels must be the same.
        """)
    end
    flatten_layers(
        [
            make(bp.expansion, in_chs),
            make(bp.depthwise, ksize, mid_chs),
            make(bp.excitation, mid_chs),
            make(bp.projection, mid_chs => out_chs)
        ]
    )
end
