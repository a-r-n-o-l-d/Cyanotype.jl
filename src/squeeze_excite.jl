using Flux: unsqueeze

@cyanotype (
"""

"""
) (
struct SqueezeExcite{A,GA}
    @volumetric
    @activation(Flux.relu)
    gate_activation::GA = Flux.sigmoid
    reduction::Int
end
)

function make(bp::SqueezeExcite; channels)
    mid_chs = channels ÷ reduction
    layers = Chain(GlobalMeanPool(),
                   flatten,
                   Dense(channels=>mid_chs, bp.activation),
                   Dense(mid_chs=>channels, bp.gate_activation),
                   _unsqueeze(bp.volumetric))
    SkipConnection(layers, .*)
end

function _unsqueeze(volumetric)
    unsq2(x) = unsqueeze(unsqueeze(x; dims = 1); dims = 1)
    if volumetric
        unsq3(x) = unsqueeze(unsq2(x); dims = 1)
    else
        unsq2
    end
end
