
@cyanotype begin
    """
    BpDConv(; kwargs)

    Describes a convolutionnal module formed by two successive convolutionnal modules.
    """
    struct BpDConv{C1<:AbstractBpConv,
                             C2<:AbstractBpConv} <: AbstractBpConv
        @volume #enlever
        conv1::C1 #conv1
        conv2::C2 = conv1 #conv2
    end
end

# channels::Pair in_chs=>out_chs out_chs=>out_chs
# channels::NTuple{3} in_chs=>mid_chs mid_chs=>out_chs
function make(bp::BpDConv, ksize, channels)
    # convolution1.vol == convolution2.vol || error("")
    c1 = spread(bp.conv1; vol = bp.volume) #cyanotype(bp.convolution1; vol = bp.volume)
    c2 = spread(bp.conv2; vol = bp.volume) #cyanotype(bp.convolution2; vol = bp.volume)
    in_chs, mid_chs, out_chs = channels
    [
        make(c1, ksize, in_chs=>mid_chs),
        make(c2, ksize, mid_chs=>out_chs)
    ] |> flatten_layers
end

# Peut-etre inutile
@cyanotype begin
    """
    Template describing a module with N `NConvBp` repeated.
    """
    struct BpNConv{C<:AbstractBpConv} <: AbstractBpConv
        convolution::C
        nrepeat::Int
    end
end

function make(bp::BpNConv, ksize, channels)
    layers = []
    in_chs, out_chs = channels
    for _ in 1:bp.nrepeat
        push!(layers, make(bp.convolution, ksize, in_chs=>out_chs))
        in_chs = out_chs
    end
    flatten_layers(layers)
end
