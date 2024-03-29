# To Do : add PixelShuffle

abstract type AbstractBpDownsampler end

abstract type AbstractBpUpsampler end

@cyanotype begin
    """
    """
    struct MaxDownsamplerBp <: AbstractBpDownsampler
        @volume
        wsize::Int = 2
    end
end

function make(bp::MaxDownsamplerBp)
    ws = genk(bp.wsize, bp.volume)
    MaxPool(ws)
end

@cyanotype begin
    """
    """
    struct MeanDownsamplerBp <: AbstractBpDownsampler
        @volume
        wsize::Int = 2
    end
end

function make(bp::MeanDownsamplerBp)
    ws = genk(bp.wsize, bp.volume)
    MeanPool(ws)
end

@cyanotype begin
    """
    """
    struct NearestUpsamplerBp <: AbstractBpUpsampler
        @volume
        scale::Int = 2
    end
end

function make(bp::NearestUpsamplerBp)
    sc = genk(bp.scale, bp.volume)
    Upsample(:nearest; scale = sc)
end

@cyanotype begin
    """
    """
    struct LinearUpsamplerBp <: AbstractBpUpsampler
        @volume
        scale::Int = 2
    end
end

function make(bp::LinearUpsamplerBp)
    if bp.volume
        Upsample(:trilinear; scale = (bp.scale, bp.scale, bp.scale))
    else
        Upsample(:bilinear; scale = (bp.scale, bp.scale))
    end
end

@cyanotype begin
    """
    """
    struct ConvTransposeUpsamplerBp{F<:Function} <: AbstractBpUpsampler
        @volume
        scale::Int = 2
        bias::Bool = true
        init::F = Flux.glorot_uniform
    end
end
# ajout kwargs (init, bias)

function make(bp::ConvTransposeUpsamplerBp, channels::Pair)
    k = genk(bp.scale, bp.volume)
    ConvTranspose(k, channels, stride=bp.scale)
end

make(bp::ConvTransposeUpsamplerBp, channels::Int) = make(bp, channels => channels)

@cyanotype constructor=false begin
    """
    """
    struct PixelShuffleUpsamplerBp <: AbstractBpUpsampler
        expansion::ChannelExpansionConvBp
        scale::Int
    end
end

function PixelShuffleUpsamplerBp(; scale=2, volume=false, normalization=BatchNormBp(), kwargs...)
    e = volume ? scale^3 : scale^2
    expansion = ChannelExpansionConvBp(; expansion=e, volume=volume, normalization=normalization, kwargs...)
    PixelShuffleUpsamplerBp(expansion, scale)
end

function make(bp::PixelShuffleUpsamplerBp, channels)
    [make(bp.expansion, channels), PixelShuffle(bp.scale)] |> flatten_layers
end
