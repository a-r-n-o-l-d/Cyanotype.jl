# https://arxiv.org/pdf/1807.06521.pdf
# https://github.com/Jongchan/attention-module/blob/5d3a54af0f6688bedca3f179593dff8da63e8274/MODELS/cbam.py#L26
@cyanotype constructor=false begin
    """

    """
    struct BpChannelAttention{GA<:Function}
        reduction::Int
        shared_mlp::BpDoubleConv #{BpPointwiseConv,BpPointwiseConv}
        gate_activation::GA = sigmoid
    end
end

function BpChannelAttention(; reduction, activation=relu, gate_activation=sigmoid, kwargs...)
    BpChannelAttention(
        reduction,
        BpDoubleConv(
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
