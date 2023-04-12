abstract type AbstractBpClassifier <: AbstractBpConv end

@cyanotype constructor=false begin
    """
    """
    struct BpPixelClassifier <: AbstractBpClassifier
        #@volume
        nclasses::Int
        convolution::BpPointwiseConv
    end
end

function BpPixelClassifier(; nclasses, activation=nclasses > 2 ? identity : sigmoid, kwargs...)
    #act = nclasses > 2 ? identity : sigmoid
    conv = BpPointwiseConv(; activation=activation, kwargs...)
    BpPixelClassifier(nclasses, conv)
end

function make(bp::BpPixelClassifier, channels)
    k = genk(1, bp.convolution.conv.volume)
    layers = []
    if bp.nclasses > 2
        push!(layers, make(bp.convolution, channels => bp.nclasses))
        push!(layers, chsoftmax)
        # x -> softmax(x; dims=length(k))
        # chcat(x...) = cat(x...; dims=(x[1] |> size |> length) - 1)
        # chsoftmax(x) = softmax(x; dims=ndims(x) - 1)
        #[Conv(k, channels => bp.nclasses), x -> softmax(x; dims = length(k))]
    else #if bp.nclasses == 2
        push!(layers, make(bp.convolution, channels => 1))
        #[Conv(k, channels=>1, sigmoid)]
    end #else Conv(k, channels=>1, sigmoid)
    flatten_layers(layers)
end
