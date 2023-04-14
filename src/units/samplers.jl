# To Do : add PixelShuffle

abstract type AbstractBpDownsampler end

abstract type AbstractBpUpsampler end

@cyanotype begin
    """
    """
    struct BpMaxDownsampler <: AbstractBpDownsampler
        @volume
        wsize::Int = 2
    end
end

function make(bp::BpMaxDownsampler)
    ws = genk(bp.wsize, bp.volume)
    MaxPool(ws)
end

@cyanotype begin
    """
    """
    struct BpMeanDownsampler <: AbstractBpDownsampler
        @volume
        wsize::Int = 2
    end
end

function make(bp::BpMeanDownsampler)
    ws = genk(bp.wsize, bp.volume)
    MeanPool(ws)
end

@cyanotype begin
    """
    """
    struct BpNearestUpsamplers <: AbstractBpUpsampler
        @volume
        scale::Int = 2
    end
end

function make(bp::BpNearestUpsamplers)
    sc = genk(bp.scale, bp.volume)
    Upsample(:nearest; scale = sc)
end

@cyanotype begin
    """
    """
    struct BpLinearUpsampler <: AbstractBpUpsampler
        @volume
        scale::Int = 2
    end
end

function make(bp::BpLinearUpsampler)
    if bp.volume
        Upsample(:trilinear; scale = (bp.scale, bp.scale, bp.scale))
    else
        Upsample(:bilinear; scale = (bp.scale, bp.scale))
    end
end

@cyanotype begin
    """
    """
    struct BpConvTransposeUpsampler <: AbstractBpUpsampler
        @volume
        scale::Int = 2
    end
end

function make(bp::BpConvTransposeUpsampler, channels::Pair)
    k = genk(bp.scale, bp.volume)
    ConvTranspose(k, channels, stride=bp.scale)
end

make(bp::BpConvTransposeUpsampler, channels::Int) = make(bp, channels => channels)

@cyanotype constructor=false begin
    """
    """
    struct BpPixelShuffleUpsampler <: AbstractBpUpsampler
        expansion::BpChannelExpansionConv
        scale::Int
    end
end

function BpPixelShuffleUpsampler(; scale=2, volume=false, normalization=BpBatchNorm(), kwargs...)
    e = volume ? scale^3 : scale^2
    expansion = BpChannelExpansionConv(; expansion=e, volume=volume, normalization=normalization, kwargs...)
    BpPixelShuffleUpsampler(expansion, scale)
end

function make(bp::BpPixelShuffleUpsampler, channels)
    [make(bp.expansion, channels), PixelShuffle(bp.scale)] |> flatten_layers
end
