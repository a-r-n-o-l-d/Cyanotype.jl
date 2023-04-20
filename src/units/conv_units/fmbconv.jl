@cyanotype constructor=false begin
    """

    """
    struct BpFusedMbConv{P<:Union{Nothing,PointwiseConvBp}} <: AbstractConvBp
        skip::Bool
        ch_expansion::Int
        convolution::ConvBp
        #dropout
        projection::P # nothing if ch_expansion == 1
    end
end

BpFusedMbConv(; stride, ch_expansion, skip=(stride == 1), activation=relu,
                         normalization=BatchNormBp(activation=activation),
                         kwargs...) = BpFusedMbConv(
    skip,
    ch_expansion,
    ConvBp(; stride=stride, activation=activation, normalization=normalization, kwargs...),
    ch_expansion <= 1 ? nothing : PointwiseConvBp(; normalization=normalization, kwargs...)
)

function make(bp::BpFusedMbConv, ksize, channels)
    in_chs, out_chs = channels
    mid_chs = in_chs * bp.ch_expansion
    if bp.skip && in_chs !== out_chs
        error("""
        If a 'BpFusedMbConv' have a skip connection defined, the number fo input channels and
        output channels must be the same.
        """)
    end
    layers = flatten_layers(
        [
            make(bp.convolution, ksize, in_chs => mid_chs),
            make(bp.projection, mid_chs => out_chs)
        ]
    )
    bp.skip ? SkipConnection(Chain(layers...), +) : layers
end
