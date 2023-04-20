# https://arxiv.org/pdf/1807.06521.pdf
# https://github.com/Jongchan/attention-module/blob/5d3a54af0f6688bedca3f179593dff8da63e8274/MODELS/cbam.py#L26
@cyanotype constructor=false begin
    """

    """
    struct BpChannelAttention{GA<:Function} <: AbstractConvBp
        reduction::Int
        shared_mlp::DoubleConvBp{BpPointwiseConv,BpPointwiseConv}
        gate_activation::GA
    end
end

function BpChannelAttention(; reduction, activation=relu, gate_activation=sigmoid,
                              kwargs...)
    BpChannelAttention(
        reduction,
        DoubleConvBp(
            conv1=BpPointwiseConv(; activation=activation, kwargs...),
            conv2=BpPointwiseConv(; activation=identity, kwargs...)
        ),
        gate_activation
    )
end

function make(bp::BpChannelAttention, channels)
    mid_chs = channels ÷ bp.reduction
    shared_mlp = Chain(make(bp.shared_mlp, (channels, mid_chs, channels))...)
    SkipConnection(
        Chain(
            Parallel(
                +,
                Chain(GlobalMeanPool(), shared_mlp),
                Chain(GlobalMaxPool(), shared_mlp)
            ),
            bp.gate_activation
        ),
        .*
    )
end

@cyanotype constructor=false begin
    """

    """
    struct BpSpatialAttention <: AbstractConvBp
        convolution::ConvBp
    end
end

BpSpatialAttention(; gate_activation=sigmoid, kwargs...) = BpSpatialAttention(
    ConvBp(; activation=gate_activation, kwargs...)
)

make(bp::BpSpatialAttention, ksize) = SkipConnection(
    Chain(
        Parallel(chcat, chmeanpool, chmaxpool),
        flatten_layers(make(bp.convolution, ksize, 2 => 1))...
    ),
    .*
)

@cyanotype constructor=false begin
    """

    """
    struct BpCBAM <: AbstractConvBp
        channel_gate::BpChannelAttention
        spatial_gate::BpSpatialAttention
    end
end

BpCBAM(; reduction, activation=relu, gate_activation=sigmoid, kwargs...) = BpCBAM(
    BpChannelAttention(;
        reduction=reduction,
        activation=activation,
        gate_activation=gate_activation,
        kwargs...
    ),
    BpSpatialAttention(;
        gate_activation=gate_activation,
        kwargs...
    )
)

make(bp::BpCBAM, ksize, channels) = flatten_layers(
    [
        make(bp.channel_gate, channels),
        make(bp.spatial_gate, ksize)
    ]
)
