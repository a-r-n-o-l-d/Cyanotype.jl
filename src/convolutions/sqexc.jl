@cyanotype constructor=false begin
    """
    https://arxiv.org/pdf/1709.01507.pdf
    """
    struct BpSqueezeExcitation
        reduction::Int
        conv1::BpPointwiseConv
        conv2::BpPointwiseConv
    end
end

function BpSqueezeExcitation(; activation=relu, gate_activation=sigmoid, reduction, kwargs...)
    # Verifier que kwargs ne contient pas activation
    haskey(kwargs, :activation) && error(
        """
        """)
    BpSqueezeExcitation(
        reduction,
        BpPointwiseConv(activation=activation, kwargs...),
        BpPointwiseConv(activation=gate_activation, kwargs...)
    )
end

function make(bp::BpSqueezeExcitation, channels)
    mid_chs = channels ÷ bp.reduction
    layers = [
        GlobalMeanPool(),
        make(bp.conv1, channels => mid_chs),
        make(bp.conv2, mid_chs => channels),
    ] |> flatten_layers
    SkipConnection(Chain(layers...), .*)
end

#=
function _unsqueeze(volumetric)
    unsq2(x) = unsqueeze(unsqueeze(x; dims = 1); dims = 1)
    if volumetric
        unsq3(x) = unsqueeze(unsq2(x); dims = 1)
    else
        unsq2
    end
end
=#
