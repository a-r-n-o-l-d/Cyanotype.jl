abstract type AbstractBpClassifier end #<: AbstractConvBp

@cyanotype constructor=false begin
    """
    """
    struct PixelClassifierBp <: AbstractConvBp
        nclasses
        convolution
    end
end

function PixelClassifierBp(; nclasses, act=nclasses > 2 ? identity : sigmoid, kwargs...)
    conv = PointwiseConvBp(; act=act, kwargs...)
    PixelClassifierBp(nclasses, conv)
end

function make(bp::PixelClassifierBp, channels)
    k = genk(1, bp.convolution.conv.vol)
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

@cyanotype begin
    """
    """
    struct LabelClassifierBp
        nclasses#::Int
        dropout#=::Real=# = 0.0
    end
end

make(bp::LabelClassifierBp, channels) = flatten_layers(
    [
        GlobalMeanPool(),
        flatten,
        iszero(bp.dropout) ? identity : Dropout(bp.dropout),
        Dense(channels => bp.nclasses, bp.nclasses == 1 ? sigmoid : identity),
        bp.nclasses == 1 ? identity : softmax
    ]
)
