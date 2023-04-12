@cyanotype constructor=false begin
    """

    """
    struct BpMBConv <: AbstractBpConv
        skip::Bool
        #dropout
        expansion::BpChannelExpansionConv
        depthwise::BpDepthwiseConv
        excitation::BpSqueezeExcitation
        projection::BpPointwiseConv
    end
end

function BpMBConv(; stride, ch_expansion, se_reduction, skip=stride == 1, activation=relu,
                  normalization=BpBatchNorm(activation=activation), kwargs...)

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

    projection = BpPointwiseConv(; normalization=normalization, kwargs...)

    BpMBConv(skip, expansion, depthwise, excitation, projection)
end

function make(bp::BpMBConv, ksize, channels)
    in_chs, out_chs = channels
    mid_chs = in_chs * bp.expansion.expansion
    if bp.skip
        in_chs == out_chs || error("""
        If a 'BpMBConv' have a skip connection defined, the number fo input channels and
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
