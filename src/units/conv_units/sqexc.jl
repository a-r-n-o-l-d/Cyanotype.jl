@cyanotype constructor=false begin
    """
    https://arxiv.org/pdf/1709.01507.pdf
    """
    struct SqueezeExcitationBp
        @volume
        reduction
        conv1
        conv2
    end
end

function SqueezeExcitationBp(; vol=false, act=relu, gate_act=sigmoid, reduction, kwargs...)
    # Verifier que kwargs ne contient pas activation
    haskey(kwargs, :act) && error(
        """
        pouet pouet
        """)
    SqueezeExcitationBp(
        vol,
        reduction,
        PointwiseConvBp(; act=act, kwargs...),
        PointwiseConvBp(; act=gate_act, kwargs...)
    )
end

function make(bp::SqueezeExcitationBp, channels)
    #if bp.reduction == 1
    #    identity
    #else
        mid_chs = max(1, channels ÷ bp.reduction)
        layers = flatten_layers(
            [
                GlobalMeanPool(), #AdaptiveMeanPool(genk(1, bp.vol)), #
                make(bp.conv1, channels => mid_chs),
                make(bp.conv2, mid_chs => channels)
            ]
        )
        SkipConnection(Chain(layers...), .*)
    #end
end
