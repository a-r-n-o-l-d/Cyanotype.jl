# to refactor
@cyanotype begin
    """
    DoubleConvBp(; kwargs)

    Describes a convolutionnal module formed by two successive convolutionnal modules.
    """
    struct DoubleConvBp <: AbstractConvBp
        conv1         #firstconv
        conv2 = conv1 #secondconv
    end
end

# channels::Pair in_chs=>out_chs out_chs=>out_chs
# channels::NTuple{3} in_chs=>mid_chs mid_chs=>out_chs
function make(bp::DoubleConvBp, ksize, channels::NTuple{3})
    # convolution1.vol == convolution2.vol || error("")
    #c1 = spread(bp.conv1; vol = bp.volume) #cyanotype(bp.convolution1; vol = bp.volume)
    #c2 = spread(bp.conv2; vol = bp.volume) #cyanotype(bp.convolution2; vol = bp.volume)
    in_chs, mid_chs, out_chs = channels
    if isnothing(bp.conv2)
        flatten_layers(make(bp.conv1, ksize, in_chs=>out_chs))
    else
        [
            make(bp.conv1, ksize, in_chs=>mid_chs),
            make(bp.conv2, ksize, mid_chs=>out_chs)
        ] |> flatten_layers
    end
end

#=
function make(bp::DoubleConvBp{<:Nothing}, ksize, channels::NTuple{3})
    in_chs, _, out_chs = channels
    flatten_layers(make(bp.conv1, ksize, in_chs=>out_chs))
end
=#

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

function make(bp::DoubleConvBp, ksize, channels::Int)
    make(bp, ksize, (channels, channels, channels))
    #=flatten_layers(
        [
            make(bp.conv1, ksize, channels),
            make(bp.conv2, ksize, channels)
        ]
    )=#
end

# Peut-etre inutile
@cyanotype begin
    """
    Template describing a module with N `NConvBp` repeated.
    """
    struct NConvBp <: AbstractConvBp
        convolutions
    end
end

function make(bp::NConvBp, ksize, channels::NTuple{3})
    layers = []
    in_chs, mid_chs, out_chs = channels
    for (i, c) in enumerate(bp.convolutions)
        if i == 1
            push!(layers, make(c, ksize, in_chs=>mid_chs))
        elseif i == length(bp.convolutions)
            push!(layers, make(c, ksize, mid_chs=>out_chs))
        else
            push!(layers, make(c, ksize, mid_chs=>mid_chs))
        end
    end
    flatten_layers(layers)
end

function make(bp::NConvBp, ksize, channels::Pair)
    in_chs, out_chs = channels
    make(bp, ksize, (in_chs, out_chs, out_chs))
end

make(bp::NConvBp, ksize, channels::Int) = make(bp, ksize, (channels, channels, channels))
