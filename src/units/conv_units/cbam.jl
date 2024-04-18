# https://arxiv.org/pdf/1807.06521.pdf
# https://github.com/Jongchan/attention-module/blob/5d3a54af0f6688bedca3f179593dff8da63e8274/MODELS/cbam.py#L26
@cyanotype constructor=false begin
    """

    """
    struct ChannelAttentionBp <: AbstractConvBp
        reduc
        mlp
        gate_act
    end
end

function ChannelAttentionBp(; reduc, act=relu, gate_act=sigmoid, kwargs...)
    ChannelAttentionBp(
        reduc,
        DoubleConvBp(
            conv1=PointwiseConvBp(; act=act, kwargs...),
            conv2=PointwiseConvBp(; act=identity, kwargs...)
        ),
        gate_act
    )
end

function make(bp::ChannelAttentionBp, channels::Int)
    mid_chs = channels ÷ bp.reduc
    shared_mlp = Chain(make(bp.mlp, (channels, mid_chs, channels))...)
    SkipConnection(
        Chain(
            Parallel(
                +,
                Chain(GlobalMeanPool(), shared_mlp),
                Chain(GlobalMaxPool(), shared_mlp)
            ),
            bp.gate_act
        ),
        .*
    )
end

make(bp::ChannelAttentionBp, channels::Pair) = make(bp, first(channels))

@cyanotype constructor=false begin
    """

    """
    struct SpatialAttentionBp <: AbstractConvBp
        conv
    end
end

SpatialAttentionBp(; gate_act=sigmoid, kwargs...) = SpatialAttentionBp(
    ConvBp(; act=gate_act, kwargs...)
)

make(bp::SpatialAttentionBp, ksize) = SkipConnection(
    Chain(
        Parallel(chcat, chmeanpool, chmaxpool),
        flatten_layers(make(bp.conv, ksize, 2 => 1))...
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

CBAMBp(; reduc, act=relu, gate_act=sigmoid, kwargs...) = CBAMBp(
    ChannelAttentionBp(;
        reduc=reduc,
        act=act,
        gate_act=gate_act,
        kwargs...
    ),
    SpatialAttentionBp(;
        gate_act=gate_act,
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

ResCBAMBp(; reduc, act=relu, gate_act=sigmoid, kwargs...) = ResCBAMBp(
    CBAMBp(;
        reduc=reduc,
        act=act,
        gate_act=gate_act,
        kwargs...
    )
)

make(bp::ResCBAMBp, ksize, channels) = SkipConnection(
    Chain(make(bp.cbam, ksize, channels)...), +
)
