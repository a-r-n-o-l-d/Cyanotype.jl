@cyanotype constructor=false begin
    """

    """
    struct BpMBConv <: AbstractBpConv
        skip::Bool
        #dropout
        expansion::BpChannelExpansion
        depthwise::BpDepthwiseConv
        excitation::BpSqueezeExcitation
        projection::BpPointwiseConv
    end
end

function BpMBConv(; stride, ch_expansion, se_reduction, skip=stride == 1, activation=swish,
                  normalization=BpBatchNorm(activation=activation), kwargs...)

    expansion = BpChannelExpansion(activation=activation,
                                   expansion=ch_expansion,
                                   normalization=normalization)

    depthwise = BpDepthwiseConv(activation=activation,
                                stride=stride,
                                normalization=normalization)

    excitation = BpSqueezeExcitation(activation=activation,
                                     gate_activation=hardσ,
                                     reduction=se_reduction)

    projection = BpPointwiseConv(normalization=normalization)

    BpMBConv(skip, expansion, depthwise, excitation, projection)
end

function make(bp::BpMBConv; ksize, channels)
    in_chs, out_chs = channels
    mid_chs = in_chs * bp.expansion.expansion
    [
        make(bp.expansion, in_chs),
        make(bp.depthwise, ksize, mid_chs),
        make(bp.excitation; mid_chs),
        make(bp.projection; mid_chs => out_chs)
    ] |> flatten_layers
end
