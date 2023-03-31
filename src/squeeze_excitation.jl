using Flux: unsqueeze, flatten

@cyanotype begin
    """
    https://arxiv.org/pdf/1709.01507.pdf
    """
    struct BpSqueezeExcitation{A,GA}
        @volumetric
        @activation(Flux.relu)
        gate_activation::GA = Flux.sigmoid
        reduction::Int #reduction_ratio
        #=
        dconvolution::C = DoubleConvolution(; convolution1 = Convolution(; activation = relu),
        convolution2 = Convolution(; activation = sigmoid))
        =#
    end
end

function make(bp::BpSqueezeExcitation; channels)
    mid_chs = channels ÷ bp.reduction
    #=
    k = bp.volumetric ? (1, 1, 1) : (1, 1)
    layers = [
        AdaptiveMeanPool(k),
        make(bp.dconv; ksize = k, channels = (channels, mid_chs, channels)),
    ]
    layers = [
        GlobalMeanPool(),
        Conv(k, channels=>mid_chs, bp.activation),
        Conv(k, channels=>mid_chs, bp.gate_activation)
    ]
    =#
    layers = Chain(GlobalMeanPool(),
                   flatten,
                   Dense(channels=>mid_chs, bp.activation),
                   Dense(mid_chs=>channels, bp.gate_activation),
                   _unsqueeze(bp.vol))
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
