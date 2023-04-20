include("effnetstage.jl")

@cyanotype constructor=false begin
    """

    """
    struct EfficientNetBp{N,
                          S<:Union{Nothing,AbstractConvBp},
                          B<:NTuple{N,EfficientNetStageBp},
                          H<:Union{Nothing,AbstractConvBp},
                          T<:Union{Nothing,LabelClassifierBp}}
        inchannels::Int = 3
        headchannels::Int = 1280
        stem::S
        backbone::B
        head::H
        top::T
    end
end

function EfficientNetBp(config; inchannels=3, headchannels=1280, nclasses,
                        include_stem=true, include_head=true, include_top=true) #activation
    stem = if include_stem
        ConvBp(; activation=swish, normalization=BatchNormBp(), stride=2)
    else
        nothing
    end
    head = if include_head
        ConvBp(; activation=swish, normalization=BatchNormBp(), stride=2)
    else
        nothing
    end
    top = if include_top
        LabelClassifierBp(nclasses=nclasses)
    else
        nothing
    end
    EfficientNetBp(inchannels, headchannels, stem, _effnet_backbone(config), head, top)
end

function make(bp::EfficientNetBp)
    layers = []
    out_chs = first(bp.backbone).outchannels
    push!(layers, make(bp.stem, 3, bp.inchannels => out_chs))
    for s in bp.backbone
        push!(layers, make(s, out_chs))
        out_chs = s.outchannels
    end
    push!(layers, make(bp.head, 3, out_chs => bp.headchannels))
    flatten_layers(layers)
end

############################################################################################
#                                   INTERNAL FUNCTIONS                                     #
############################################################################################

function _effnet_backbone(config)
    if config ∈ [:b0, :b1, :b2, :b3, :b4, :b5, :b6, :b7, :b8]
        _effnetv1_backbone(config)
    elseif config ∈ [:small, :medium, :large, :xlarge]
        _effnetv2_backbone(config)
    else
       error(
            """
            $config is not an accepted configuration
            """
        )
    end
end

_effnetv1_backbone(wsc, dsc) = (
    EfficientNetStageBp(MbConvBp, 3, 16, 1, 1, 1, 4, wsc, dsc),
    EfficientNetStageBp(MbConvBp, 3, 24, 6, 2, 2, 4, wsc, dsc),
    EfficientNetStageBp(MbConvBp, 5, 40, 6, 2, 2, 4, wsc, dsc),
    EfficientNetStageBp(MbConvBp, 3, 80, 6, 2, 3, 4, wsc, dsc),
    EfficientNetStageBp(MbConvBp, 5, 112, 6, 1, 3, 4, wsc, dsc),
    EfficientNetStageBp(MbConvBp, 5, 192, 6, 2, 4, 4, wsc, dsc),
    EfficientNetStageBp(MbConvBp, 3, 320, 6, 1, 1, 4, wsc, dsc)
)

function _effnetv1_backbone(config)
    if config == :b0
        _effnetv1_backbone(1.0, 1.0)
    elseif config == :b1
        _effnetv1_backbone(1.0, 1.1)
    elseif config == :b2
        _effnetv1_backbone(1.1, 1.2)
    elseif config == :b3
        _effnetv1_backbone(1.2, 1.4)
    elseif config == :b4
        _effnetv1_backbone(1.4, 1.8)
    elseif config == :b5
        _effnetv1_backbone(1.6, 2.2)
    elseif config == :b6
        _effnetv1_backbone(1.8, 2.6)
    elseif config == :b7
        _effnetv1_backbone(2.0, 3.1)
    elseif config == :b8
        _effnetv1_backbone(2.2, 3.6)
    end
end

function _effnetv2_backbone(config)
    if config == :small
        (
            EfficientNetStageBp(FusedMbConvBp, 3, 24, 1, 1, 2),
            EfficientNetStageBp(FusedMbConvBp, 3, 48, 4, 2, 4),
            EfficientNetStageBp(FusedMbConvBp, 3, 64, 4, 2, 4),
            EfficientNetStageBp(MbConvBp, 3, 128, 4, 2, 6,  4),
            EfficientNetStageBp(MbConvBp, 3, 160, 6, 1, 9,  4),
            EfficientNetStageBp(MbConvBp, 3, 256, 6, 2, 15, 4)
        )
    elseif config == :medium
        (
            EfficientNetStageBp(FusedMbConvBp, 3, 24, 1, 1, 3),
            EfficientNetStageBp(FusedMbConvBp, 3, 48, 4, 2, 5),
            EfficientNetStageBp(FusedMbConvBp, 3, 80, 4, 2, 5),
            EfficientNetStageBp(MbConvBp, 3, 160, 4, 2, 7,  4),
            EfficientNetStageBp(MbConvBp, 3, 176, 6, 1, 14, 4),
            EfficientNetStageBp(MbConvBp, 3, 304, 6, 2, 18, 4),
            EfficientNetStageBp(MbConvBp, 3, 512, 6, 1, 5,  4)
        )
    elseif config == :large
        (
            EfficientNetStageBp(FusedMbConvBp, 3, 32, 1, 1, 4),
            EfficientNetStageBp(FusedMbConvBp, 3, 64, 4, 2, 7),
            EfficientNetStageBp(FusedMbConvBp, 3, 96, 4, 2, 7),
            EfficientNetStageBp(MbConvBp, 3, 192, 4, 2, 10, 4),
            EfficientNetStageBp(MbConvBp, 3, 224, 6, 1, 19, 4),
            EfficientNetStageBp(MbConvBp, 3, 384, 6, 2, 25, 4),
            EfficientNetStageBp(MbConvBp, 3, 640, 6, 1, 7,  4)
        )
    elseif config == :xlarge
        (
            EfficientNetStageBp(FusedMbConvBp, 3, 32, 1, 1, 4),
            EfficientNetStageBp(FusedMbConvBp, 3, 64, 4, 2, 8),
            EfficientNetStageBp(FusedMbConvBp, 3, 96, 4, 2, 8),
            EfficientNetStageBp(MbConvBp, 3, 192, 4, 2, 16, 4),
            EfficientNetStageBp(MbConvBp, 3, 384, 6, 1, 24, 4),
            EfficientNetStageBp(MbConvBp, 3, 512, 6, 2, 32, 4),
            EfficientNetStageBp(MbConvBp, 3, 768, 6, 1, 8,  4)
        )
    end
end
