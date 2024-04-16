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
    ws = genk(bp.wsize, bp.pw.vol)
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
    ws = genk(bp.wsize, bp.vol)
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
    ws = genk(bp.wsize, bp.vol)
    MeanPool(ws)
end

@cyanotype begin
    """
    """
    struct NearestUpsamplerBp <: AbstractBpUpsampler
        @volume
        scale = 2
    end
end

function make(bp::NearestUpsamplerBp)
    sc = genk(bp.scale, bp.vol)
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
    if bp.vol
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
    k = genk(bp.scale, bp.vol)
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

function PixelShuffleUpsamplerBp(; scale=2, vol=false, norm=BatchNormBp(), kwargs...)
    e = vol ? scale^3 : scale^2
    expansion = ChannelExpansionConvBp(; expansion=e, vol=vol, norm=norm, kwargs...)
    PixelShuffleUpsamplerBp(expansion, scale)
end

function make(bp::PixelShuffleUpsamplerBp, channels)
    [make(bp.expansion, channels), PixelShuffle(bp.scale)] |> flatten_layers
end
