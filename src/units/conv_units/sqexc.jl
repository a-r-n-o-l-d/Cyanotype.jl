@cyanotype constructor=false begin
    """
    https://arxiv.org/pdf/1709.01507.pdf
    """
    struct BpSqueezeExcitation
        reduction::Int
        #convolution::BpDoubleConv{BpPointwiseConv,BpPointwiseConv}
        conv1::BpPointwiseConv
        conv2::BpPointwiseConv
    end
end

function BpSqueezeExcitation(; activation=relu, gate_activation=sigmoid, reduction, kwargs...)
    # Verifier que kwargs ne contient pas activation
    haskey(kwargs, :activation) && error(
        """
        pouet pouet
        """)
    BpSqueezeExcitation(
        reduction,
        BpPointwiseConv(; activation=activation, kwargs...),
        BpPointwiseConv(; activation=gate_activation, kwargs...)
    )
end

function make(bp::BpSqueezeExcitation, channels)
    mid_chs = channels ÷ bp.reduction
    layers = flatten_layers(
        [
            GlobalMeanPool(),
            make(bp.conv1, channels => mid_chs),
            make(bp.conv2, mid_chs => channels),
        ]
    )
    SkipConnection(Chain(layers...), .*)
end
