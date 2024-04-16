# To Do : add PixelShuffle

abstract type AbstractBpDownsampler end

abstract type AbstractBpUpsampler end

@cyanotype constructor=false begin
    """
    """
    struct MeanMaxDownsamplerBp <: AbstractBpDownsampler
        pw
        wsize#=::Int=# = 2
    end
end

MeanMaxDownsamplerBp(; wsize=2, kwargs...) = MeanMaxDownsamplerBp(PointwiseConvBp(; kwargs...), wsize)

function make(bp::MeanMaxDownsamplerBp, channels)
    ws = genk(bp.wsize, bp.pw.volume)
    flatten_layers(
        [
            Parallel(chcat, MeanPool(ws), MaxPool(ws)),
            make(bp.pw, 2 * channels => channels)
        ]
    )
end

@cyanotype begin
    """
    """
    struct MaxDownsamplerBp <: AbstractBpDownsampler
        @volume
        wsize#=::Int=# = 2
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
        wsize = 2
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
        scale#=:Int=# = 2
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
        scale#=::Int=# = 2
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
    struct ConvTransposeUpsamplerBp <: AbstractBpUpsampler
        @volume
        scale = 2
        bias = true
        init = Flux.glorot_uniform
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
        expansion
        scale
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
