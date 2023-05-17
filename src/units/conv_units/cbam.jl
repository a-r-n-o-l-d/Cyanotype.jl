# https://arxiv.org/pdf/1807.06521.pdf
# https://github.com/Jongchan/attention-module/blob/5d3a54af0f6688bedca3f179593dff8da63e8274/MODELS/cbam.py#L26
@cyanotype constructor=false begin
    """

    """
    struct ChannelAttentionBp{GA<:Function} <: AbstractConvBp
        reduction::Int
        shared_mlp::DoubleConvBp{PointwiseConvBp,PointwiseConvBp}
        gate_activation::GA
    end
end

function ChannelAttentionBp(; reduction, activation=relu, gate_activation=sigmoid,
                              kwargs...)
    ChannelAttentionBp(
        reduction,
        DoubleConvBp(
            conv1=PointwiseConvBp(; activation=activation, kwargs...),
            conv2=PointwiseConvBp(; activation=identity, kwargs...)
        ),
        gate_activation
    )
end

function make(bp::ChannelAttentionBp, channels::Int)
    mid_chs = channels ÷ bp.reduction
    shared_mlp = Chain(make(bp.shared_mlp, (channels, mid_chs, channels))...)
    SkipConnection(
        Chain(
            Parallel(
                +,
                Chain(GlobalMeanPool(), shared_mlp), #AdaptiveMeanPool(genk(1, bp.volume)),
                Chain(GlobalMaxPool(), shared_mlp)
            ),
            bp.gate_activation
        ),
        .*
    )
end

#make(bp::ChannelAttentionBp, channels::Int) = make(bp,  channels => channels)

@cyanotype constructor=false begin
    """

    """
    struct SpatialAttentionBp <: AbstractConvBp
        convolution::ConvBp
    end
end

SpatialAttentionBp(; gate_activation=sigmoid, kwargs...) = SpatialAttentionBp(
    ConvBp(; activation=gate_activation, kwargs...)
)

make(bp::SpatialAttentionBp, ksize) = SkipConnection(
    Chain(
        Parallel(chcat, chmeanpool, chmaxpool),
        flatten_layers(make(bp.convolution, ksize, 2 => 1))...
    ),
    .*
)

@cyanotype constructor=false begin
    """

    """
    struct CBAMBp <: AbstractConvBp
        channel_gate::ChannelAttentionBp
        spatial_gate::SpatialAttentionBp
    end
end

CBAMBp(; reduction, activation=relu, gate_activation=sigmoid, kwargs...) = CBAMBp(
    ChannelAttentionBp(;
        reduction=reduction,
        activation=activation,
        gate_activation=gate_activation,
        kwargs...
    ),
    SpatialAttentionBp(;
        gate_activation=gate_activation,
        kwargs...
    )
)

make(bp::CBAMBp, ksize, channels) = flatten_layers(
    [
        make(bp.channel_gate, channels),
        make(bp.spatial_gate, ksize)
    ]
)
