# To Do : add PixelShuffle

abstract type AbstractBpDownSampler end

abstract type AbstractBpUpSampler end

@cyanotype begin
    """
    """
    struct BpMaxDown <: AbstractBpDownSampler
        @volume
        wsize::Int = 2
    end
end

function make(bp::BpMaxDown)
    ws = genk(bp.wsize, bp.volume)
    MaxPool(ws)
end

@cyanotype begin
    """
    """
    struct BpMeanDown <: AbstractBpDownSampler
        @volume
        wsize::Int = 2
    end
end

function make(bp::BpMeanDown)
    ws = genk(bp.wsize, bp.volume)
    MeanPool(ws)
end

@cyanotype begin
    """
    """
    struct BpNearestUp <: AbstractBpUpSampler
        @volume
        scale::Int = 2
    end
end

function make(bp::BpNearestUp)
    sc = genk(bp.scale, bp.volume)
    Upsample(:nearest; scale = sc)
end

@cyanotype begin
    """
    """
    struct BpLinearUp <: AbstractBpUpSampler
        @volume
        scale::Int = 2
    end
end

function make(bp::BpLinearUp)
    if bp.volume
        Upsample(:trilinear; scale = (bp.scale, bp.scale, bp.scale))
    else
        Upsample(:bilinear; scale = (bp.scale, bp.scale))
    end
end

@cyanotype begin
    """
    """
    struct BpConvTransposeUp <: AbstractBpUpSampler
        @volume
        scale::Int = 2
    end
end

function make(bp::BpConvTransposeUp, channels)
    k = genk(bp.scale, bp.volume)
    ConvTranspose(k, channels, stride = bp.scale)
end

@cyanotype constructor=false begin
    """
    """
    struct BpPixelShuffleUp <: AbstractBpUpSampler
        expansion::BpChannelExpansionConv
        scale::Int
    end
end

function BpPixelShuffleUp(; scale=2, volume=false, normalization=BpBatchNorm(), kwargs...)
    e = volume ? scale^3 : scale^2
    expansion = BpChannelExpansionConv(; expansion=e, volume=volume, normalization=normalization, kwargs...)
    BpPixelShuffleUp(expansion, scale)
end

function make(bp::BpPixelShuffleUp, channels)
    [make(bp.expansion, channels), PixelShuffle(bp.scale)] |> flatten_layers
end
