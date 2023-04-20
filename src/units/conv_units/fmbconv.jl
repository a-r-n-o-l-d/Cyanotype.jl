@cyanotype constructor=false begin
    """

    """
    struct BpFusedMBConv{P<:Union{Nothing,BpPointwiseConv}} <: AbstractBpConv
        skip::Bool
        ch_expansion::Int
        convolution::BpConv
        #dropout
        projection::P # nothing if ch_expansion == 1
    end
end

BpFusedMBConv(; stride, ch_expansion, skip=(stride == 1), activation=relu,
                         normalization=BpBatchNorm(activation=activation),
                         kwargs...) = BpFusedMBConv(
    skip,
    ch_expansion,
    BpConv(; stride=stride, activation=activation, normalization=normalization, kwargs...),
    ch_expansion <= 1 ? nothing : BpPointwiseConv(; normalization=normalization, kwargs...)
)

function make(bp::BpFusedMBConv, ksize, channels)
    in_chs, out_chs = channels
    mid_chs = in_chs * bp.ch_expansion
    if bp.skip && in_chs !== out_chs
        error("""
        If a 'BpFusedMBConv' have a skip connection defined, the number fo input channels and
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
