include("effnetstage.jl")

const EFFNETV1 = [:b0, :b1, :b2, :b3, :b4, :b5, :b6, :b7, :b8]

const EFFNETV2 = [:small, :medium, :large, :xlarge]

@cyanotype constructor=false begin
    """

    """
    struct EfficientNetBp
        in_chs
        st_chs
        hd_chs
        stem
        bbone
        head
        top
    end
end

function EfficientNetBp(config; in_chs=3, st_chs=32, hd_chs=1280, ncls=1000, stem=true,
                        top=true, head=top)
    # Sanity check
    _check_effnet_config(config)
    if top && !head
        error(
            """
            You must set 'include_head' to true if you want to include top layers.
            """
        )
    end
    st = stem ? ConvBp(; act=swish, norm=BatchNormBp(), stride=2) : nothing
    hd = head ? ConvBp(; act=swish, norm=BatchNormBp()) : nothing
    tp = top ? LabelClassifierBp(nclasses=ncls) : nothing
    bb = _effnet_backbone(config)
    st_chs = if config ∈ EFFNETV1
        wsc, _ = _effnetv1_scaling(config)
        _round_channels(st_chs * wsc)
    elseif config ∈ EFFNETV2
        first(bb).out_chs
    end
    EfficientNetBp(in_chs, st_chs, hd_chs, st, bb, hd, tp)
end

function make(bp::EfficientNetBp) #dropout
    drop_prob = 0.2
    block_repeats = 0
    for s in bp.bbone
        block_repeats = block_repeats + s.nrepeat + 1
    end
    dropouts = LinRange(0, drop_prob, block_repeats + 1)[1:block_repeats]
    out_chs = bp.st_chs
    stem = Chain(flatten_layers(make(bp.stem, 3, bp.in_chs => out_chs))...)
    layers = []
    block_idx = 1
    for s in bp.bbone
        dps = dropouts[block_idx:block_idx + s.nrepeat]
        push!(layers, Chain(make(s, out_chs, dropouts=dps))...)
        out_chs = s.out_chs
        block_idx = block_idx + s.nrepeat + 1
    end
    bbone = Chain(flatten_layers(layers)...)
    head = Chain(flatten_layers(make(bp.head, 1, out_chs => bp.hd_chs))...)
    top = Chain(flatten_layers(make(bp.top, bp.hd_chs))...)
    Chain(stem=stem, bbone=bbone, head=head, top=top)
end

############################################################################################
#                                   INTERNAL FUNCTIONS                                     #
############################################################################################

function _check_effnet_config(config)
    if config ∉ EFFNETV1 && config ∉ EFFNETV2
       error(
            """
            $config is not a valid configuration, 'config' must be in $EFFNETV1
            (EfficientNetV1 architecture) or in $EFFNETV2 (EfficientNetV2 architecture).
            """
        )
    end
end

function _effnet_backbone(config)
    if config ∈ EFFNETV1
        wsc, dsc = _effnetv1_scaling(config)
        _effnetv1_backbone(wsc, dsc)
    elseif config ∈ EFFNETV2
        _effnetv2_backbone(config)
    end
end

function _effnetv1_scaling(config)
    if config == :b0
        (1.0, 1.0)
    elseif config == :b1
        (1.0, 1.1)
    elseif config == :b2
        (1.1, 1.2)
    elseif config == :b3
        (1.2, 1.4)
    elseif config == :b4
        (1.4, 1.8)
    elseif config == :b5
        (1.6, 2.2)
    elseif config == :b6
        (1.8, 2.6)
    elseif config == :b7
        (2.0, 3.1)
    elseif config == :b8
        (2.2, 3.6)
    end
end

_effnetv1_backbone(wsc, dsc) = (
    EfficientNetStageBp(MbConvBp, 3, 16,  1, 1, 1, 4, wsc, dsc),
    EfficientNetStageBp(MbConvBp, 3, 24,  6, 2, 2, 4, wsc, dsc),
    EfficientNetStageBp(MbConvBp, 5, 40,  6, 2, 2, 4, wsc, dsc),
    EfficientNetStageBp(MbConvBp, 3, 80,  6, 2, 3, 4, wsc, dsc),
    EfficientNetStageBp(MbConvBp, 5, 112, 6, 1, 3, 4, wsc, dsc),
    EfficientNetStageBp(MbConvBp, 5, 192, 6, 2, 4, 4, wsc, dsc),
    EfficientNetStageBp(MbConvBp, 3, 320, 6, 1, 1, 4, wsc, dsc)
)

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
