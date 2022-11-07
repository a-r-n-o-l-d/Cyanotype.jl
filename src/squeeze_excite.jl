@cyanotype (
"""

"""
) (
struct SqueezeExcite{GA}
    @volumetric
    @activation(Flux.relu)
    gate_activation::GA = Flux.sigmoid
    reduction::Int
end
)

function make(bp::SqueezeExcite, channels)

end

function squeeze_and_excitation(chs; reduction, activation = relu, gate_activation = sigmoid)
    layers = Chain(GlobalMeanPool(),
                    flatten,
                    Dense(chs=>chs ÷ reduction, activation),
                    Dense(chs ÷ reduction=>chs, gate_activation),
                    unsqueeze(dims = 1) ∘ unsqueeze(dims = 1)) # Pour 3D : unsqueeze(dims = 1) ∘ unsqueeze(dims = 1) ∘ unsqueeze(dims = 1), plutot pipe
    SkipConnection(layers, .*)
end
