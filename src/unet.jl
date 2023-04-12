@cyanotype begin
    """

    """
    struct BpUEncoder{C<:AbstractBpConv,D<:Union{Nothing,AbstractBpDownSampler}}
        convolution::C #BpDoubleConv
        downsampler::D #nothing si stride=2
    end
end

function make(bp::BpUEncoder, ksize, channels)
    flatten_layers(
        [
            make(bp.downsampler),
            make(bp.convolution, ksize, channels)
        ]
    )
end

@cyanotype begin
    """

    """
    struct BpUDecoder{C<:AbstractBpConv,U<:AbstractBpUpSampler}
        convolution::C
        upsampler::U
    end
end

function make(bp::BpUDecoder, ksize, channels)
    flatten_layers(
        [
            make(bp.convolution, ksize, channels),
            _make(bp.upsampler, channels)
        ]
    )
end

@cyanotype begin
    """

    """
    struct BpUBridge{D<:Union{Nothing,AbstractBpDownSampler},U<:AbstractBpUpSampler} #,P<:BpPixelClassifierBp
        convolution::BpDoubleConv
        downsampler::D #nothing si stride=2
        upsampler::U
    end
end

function make(bp::BpUBridge, ksize, channels)
    flatten_layers(
        [
            make(bp.downsampler),
            make(bp.convolution, ksize, channels),
            _make(bp.upsampler, channels)
        ]
    )
end
