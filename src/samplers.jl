# To Do : add PixelShuffle

abstract type AbstractDownSamplerBp end

abstract type AbstractUpSamplerBp end

@cyanotype begin
    """
    """
    struct MaxDownSamplerBp <: AbstractDownSamplerBp
        @volume
        wsize::Int = 2
    end
end

function make(bp::MaxDownSamplerBp)
    ws = genk(bp.wsize, bp.volume)
    MaxPool(ws)
end

@cyanotype begin
    """
    """
    struct MeanDownSamplerBp <: AbstractDownSamplerBp
        @volume
        wsize::Int = 2
    end
end

function make(bp::MeanDownSamplerBp)
    ws = genk(bp.wsize, bp.volume)
    MeanPool(ws)
end

@cyanotype begin
    """
    """
    struct NearestUpSamplerBp <: AbstractUpSamplerBp
        @volume
        scale::Int = 2
    end
end

function make(bp::NearestUpSamplerBp)
    sc = genk(bp.scale, bp.volume)
    Upsample(:nearest; scale = sc)
end

@cyanotype begin
    """
    """
    struct LinearUpSamplerBp <: AbstractUpSamplerBp
        @volume
        scale::Int = 2
    end
end

function make(bp::LinearUpSamplerBp)
    if bp.volume
        Upsample(:trilinear; scale = (bp.scale, bp.scale, bp.scale))
    else
        Upsample(:bilinear; scale = (bp.scale, bp.scale))
    end
end

@cyanotype begin
    """
    """
    struct ConvUpSamplerBp <: AbstractUpSamplerBp
        @volume
        scale::Int = 2
    end
end

function make(bp::ConvUpSamplerBp; channels)
    k = genk(bp.scale, bp.volume)
    ConvTranspose(k, channels, stride = bp.scale)
end
