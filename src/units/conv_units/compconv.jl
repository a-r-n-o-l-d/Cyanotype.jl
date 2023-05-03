# to refactor
@cyanotype begin
    """
    DoubleConvBp(; kwargs)

    Describes a convolutionnal module formed by two successive convolutionnal modules.
    """
    struct DoubleConvBp{C1<:AbstractConvBp,C2<:AbstractConvBp} <: AbstractConvBp
        #@volume #enlever
        conv1::C1         #firstconv
        conv2::C2 = conv1 #secondconv
    end
end

# channels::Pair in_chs=>out_chs out_chs=>out_chs
# channels::NTuple{3} in_chs=>mid_chs mid_chs=>out_chs
function make(bp::DoubleConvBp, ksize, channels::NTuple{3})
    # convolution1.vol == convolution2.vol || error("")
    #c1 = spread(bp.conv1; vol = bp.volume) #cyanotype(bp.convolution1; vol = bp.volume)
    #c2 = spread(bp.conv2; vol = bp.volume) #cyanotype(bp.convolution2; vol = bp.volume)
    in_chs, mid_chs, out_chs = channels
    [
        make(bp.conv1, ksize, in_chs=>mid_chs),
        make(bp.conv2, ksize, mid_chs=>out_chs)
    ] |> flatten_layers
end

function make(bp::DoubleConvBp, ksize, channels::Pair)
    in_chs, out_chs = channels
    make(bp, ksize, (in_chs, out_chs, out_chs))
end

function make(bp::DoubleConvBp, channels::NTuple{3})
    # convolution1.vol == convolution2.vol || error("")
    in_chs, mid_chs, out_chs = channels
    flatten_layers(
        [
            make(bp.conv1, in_chs=>mid_chs),
            make(bp.conv2, mid_chs=>out_chs)
        ]
    )
end

# Peut-etre inutile
@cyanotype begin
    """
    Template describing a module with N `NConvBp` repeated.
    """
    struct NConvBp{C<:AbstractConvBp} <: AbstractConvBp
        convolution::C
        nrepeat::Int
    end
end

function make(bp::NConvBp, ksize, channels)
    layers = []
    in_chs, out_chs = channels
    for _ in 1:bp.nrepeat
        push!(layers, make(bp.convolution, ksize, in_chs=>out_chs))
        in_chs = out_chs
    end
    flatten_layers(layers)
end
