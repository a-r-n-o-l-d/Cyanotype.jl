@cyanotype constructor=false begin
    """

    """
    struct MbConvBp <: AbstractConvBp
        skip::Bool
        #dropout
        expansion::ChannelExpansionConvBp
        depthwise::DepthwiseConvBp
        excitation::SqueezeExcitationBp
        projection::PointwiseConvBp
    end
end

# MbConvBp(expkwargs, depkwargs, exckwargs, projkwargs; stride, ch_expansion, se_reduction, skip=stride == 1, activation=relu,
# normalization=BatchNormBp(activation=activation))
function MbConvBp(; stride, ch_expansion, se_reduction, skip=stride == 1, activation=relu,
                  normalization=BatchNormBp(activation=activation), kwargs...)

    stride ∈ [1, 2] || error("`stride` has to be 1 or 2 for `MbConvBp`")

    expansion = ChannelExpansionConvBp(; activation=activation,
                                         expansion=ch_expansion,
                                         normalization=normalization,
                                         kwargs...)

    depthwise = DepthwiseConvBp(; activation=activation,
                                  stride=stride,
                                  normalization=normalization,
                                  kwargs...)

    excitation = SqueezeExcitationBp(; activation=activation,
                                       gate_activation=hardσ,
                                       reduction=se_reduction * ch_expansion,
                                       kwargs...)

    projection = PointwiseConvBp(; normalization=normalization, kwargs...)

    MbConvBp(skip, expansion, depthwise, excitation, projection)
end

#function make(bp::MbConvBp, ksize, channels, dropout=0.0)
function make(bp::MbConvBp, ksize, channels, dropout=0) # add dropout for stochastic depth
    in_chs, out_chs = channels
    mid_chs = in_chs * bp.expansion.expansion
    #=if bp.skip && in_chs !== out_chs
        error("""
        If a 'MbConvBp' have a skip connection defined, the number fo input channels and
        output channels must be the same.
        """)
    end=#
    layers = flatten_layers(
        [
            make(bp.expansion, in_chs),
            make(bp.depthwise, ksize, mid_chs),
            make(bp.excitation, mid_chs),
            make(bp.projection, mid_chs => out_chs)
        ]
    )
    if bp.skip && in_chs == out_chs
        if iszero(dropout)
            SkipConnection(Chain(layers...), +)
        else
            d = bp.projection.conv.volume ? 5 : 4
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
    mid_chs = in_chs * bp.expansion.expansion
    flatten_layers(
        [
            make(bp.expansion, in_chs),
            make(bp.depthwise, ksize, mid_chs),
            make(bp.excitation, mid_chs),
            make(bp.projection, mid_chs => out_chs)
        ]
    )
end
