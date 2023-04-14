# https://arxiv.org/pdf/1807.06521.pdf
# https://github.com/Jongchan/attention-module/blob/5d3a54af0f6688bedca3f179593dff8da63e8274/MODELS/cbam.py#L26
@cyanotype constructor=false begin
    """

    """
    struct BpChannelAttention{GA<:Function}
        reduction::Int
        shared_mlp::BpDoubleConv{BpPointwiseConv,BpPointwiseConv}
        gate_activation::GA
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

@cyanotype constructor=false begin
    """

    """
    struct BpSpatialAttention #{GA<:Function}
        convolution::BpPointwiseConv
        #gate_activation::GA
    end
end

BpSpatialAttention(; gate_activation=sigmoid, kwargs...) = BpSpatialAttention(
    BpPointwiseConv(; activation=gate_activation, kwargs...)
)

chmeanpool(x) = mean(x; dims=ndims(x) - 1)

chmaxpool(x) = maximum(x; dims=ndims(x) - 1)

function make(bp::BpSpatialAttention)
    SkipConnection(
        Chain(
            Parallel(chcat, chmeanpool, chmaxpool),
            flatten_layers(make(bp.convolution, 2 => 1))...
        ),
        .*
    )
end
