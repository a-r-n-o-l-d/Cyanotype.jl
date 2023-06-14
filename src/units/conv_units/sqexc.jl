@cyanotype constructor=false begin
    """
    https://arxiv.org/pdf/1709.01507.pdf
    """
    struct SqueezeExcitationBp
        @volume
        reduction::Int
        #convolution::DoubleConvBp{PointwiseConvBp,PointwiseConvBp}
        conv1::PointwiseConvBp
        conv2::PointwiseConvBp
    end
end

function SqueezeExcitationBp(; volume =false, activation=relu, gate_activation=sigmoid, reduction, kwargs...)
    # Verifier que kwargs ne contient pas activation
    haskey(kwargs, :activation) && error(
        """
        pouet pouet
        """)
    SqueezeExcitationBp(
        volume,
        reduction,
        PointwiseConvBp(; activation=activation, kwargs...),
        PointwiseConvBp(; activation=gate_activation, kwargs...)
    )
end

function make(bp::SqueezeExcitationBp, channels)
    #if bp.reduction == 1
    #    identity
    #else
        mid_chs = max(1, channels ÷ bp.reduction)
        layers = flatten_layers(
            [
                GlobalMeanPool(), #AdaptiveMeanPool(genk(1, bp.volume)), #
                make(bp.conv1, channels => mid_chs),
                make(bp.conv2, mid_chs => channels)
            ]
        )
        SkipConnection(Chain(layers...), .*)
    #end
end
