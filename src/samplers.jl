abstract type AbstractDownSamplerBp end

abstract type AbstractUpSamplerBp end

@cyanotype begin
    """
    """
    struct MaxDownSamplerBp <: AbstractDownSamplerBp
        @volumetric
        wsize::Int = 2
    end
end

function make(bp::MaxDownSamplerBp)
    ws = genk(bp.wsize, bp.vol)
    MaxPool(ws)
end

@cyanotype begin
    """
    """
    struct MeanDownSamplerBp <: AbstractDownSamplerBp
        @volumetric
        wsize::Int = 2
    end
end

function make(bp::MeanDownSamplerBp)
    ws = genk(bp.wsize, bp.vol)
    MeanPool(ws)
end

@cyanotype begin
    """
    """
    struct NearestUpSamplerBp <: AbstractUpSamplerBp
        @volumetric
        scale::Int = 2
    end
end

function make(bp::NearestUpSamplerBp)
    sc = genk(bp.scale, bp.vol)
    Upsample(:nearest; scale = sc)
end

@cyanotype begin
    """
    """
    struct LinearUpSamplerBp <: AbstractUpSamplerBp
        @volumetric
        scale::Int = 2
    end
end

function make(bp::LinearUpSamplerBp)
    if bp.vol
        Upsample(:trilinear; scale = (bp.scale, bp.scale, bp.scale))
    else
        Upsample(:bilinear; scale = (bp.scale, bp.scale))
    end
end

@cyanotype begin
    """
    """
    struct ConvUpSamplerBp <: AbstractUpSamplerBp
        @volumetric
        scale::Int = 2
    end
end

function make(bp::ConvUpSamplerBp; channels)
    k = genk(bp.scale, bp.vol)
    ConvTranspose(k, channels, stride = bp.scale)
end
