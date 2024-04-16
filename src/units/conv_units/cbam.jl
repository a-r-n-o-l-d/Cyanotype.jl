# https://arxiv.org/pdf/1807.06521.pdf
# https://github.com/Jongchan/attention-module/blob/5d3a54af0f6688bedca3f179593dff8da63e8274/MODELS/cbam.py#L26
@cyanotype constructor=false begin
    """

    """
    struct ChannelAttentionBp <: AbstractConvBp
        reduction
        shared_mlp
        gate_activation
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
                Chain(GlobalMeanPool(), shared_mlp),
                Chain(GlobalMaxPool(), shared_mlp)
            ),
            bp.gate_activation
        ),
        .*
    )
end

make(bp::ChannelAttentionBp, channels::Pair) = make(bp, first(channels))

@cyanotype constructor=false begin
    """

    """
    struct SpatialAttentionBp <: AbstractConvBp
        convolution
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
        channel_gate
        spatial_gate
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

@cyanotype constructor=false begin
    """

    """
    struct ResCBAMBp <: Cyanotype.AbstractConvBp
        cbam
    end
end

ResCBAMBp(; reduction, activation=relu, gate_activation=sigmoid, kwargs...) = ResCBAMBp(
    CBAMBp(;
        reduction=reduction,
        activation=activation,
        gate_activation=gate_activation,
        kwargs...
    )
)

make(bp::ResCBAMBp, ksize, channels) = SkipConnection(Chain(make(bp.cbam, ksize, channels)...), +)
