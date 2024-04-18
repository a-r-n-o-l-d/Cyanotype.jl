@cyanotype constructor=false begin
    """
    aka Hybrid Dilated Convolution
    [paper](@ref https://doi.org/10.1109/WACV.2018.00163)
    [example](@ref https://doi.org/10.1016/j.image.2019.115664)
    [example](@ref https://doi-org/10.1109/ICMA54519.2022.9855903)
    """
    struct HybridAtrouConvBp <: AbstractConvBp #HybridAtrouConvBp
        dilar
        conv
    end
end

HybridAtrouConvBp(; dilar=(1, 2, 3), norm=BatchNormBp(), kwargs...) = HybridAtrouConvBp(
    dilar,
    ConvBp(; norm=norm, kwargs...)
)

function make(bp::HybridAtrouConvBp, ksize, channels::Pair)
    check_dilation_rates(ksize, bp.dilar) || error("Invalid dilation rates.")
    layers = []
    in_chs, out_chs = channels
    for dr in bp.dilar
        c = cyanotype(bp.conv; dila=dr)
        push!(layers, flatten_layers(make(c, ksize, in_chs=>out_chs)))
        in_chs = out_chs
    end
    flatten_layers(layers)
end

make(bp::HybridAtrouConvBp, ksize, channels::Int) = make(bp, ksize, channels => channels)

# https://arxiv.org/pdf/1702.08502.pdf
# DOI 10.1109/WACV.2018.00163
function check_dilation_rates(k, dr)
    issorted(dr) || error("Dilation rates must be increasing.")
    M = dr[end]
    i = length(dr)
    while i > 2
        i = i - 1
        M = max(M - 2 * dr[i], M - 2 * (M - dr[i]), dr[i])
    end
    M <= k
end
