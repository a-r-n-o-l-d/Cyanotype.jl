@cyanotype begin
    """

    """
    struct BpEfficientNetStage{C<:AbstractBpConv,R1<:Union{Nothing,Real},R2<:Union{Nothing,Real}} <: AbstractBpConv
        ksize::Int
        outchannels::Int
        nrepeat::Int
        convolution::C # BpMBConv(; stride, ch_expansion, se_reduction, activation)
        widthscaling::R1 = nothing
        depthscaling::R2 = nothing
    end
end

function make(bp::BpEfficientNetStage, channels::Int)
    in_chs = _round_channels(channels)
    out_chs = isnothing(bp.widthscaling) ? _round_channels(bp.outchannels) : _round_channels(bp.outchannels * bp.widthscaling)
    layers = []
    push!(layers, make(bp.convolution, bp.ksize, in_chs => out_chs))
    nrepeat = isnothing(bp.depthscaling) ? bp.nrepeat - 1 : ceil(Int, bp.nrepeat * bp.depthscaling) - 1
    for _ in 1:nrepeat
        conv = spread(bp.convolution; stride = 1, skip=true)
        push!(layers, make(conv, bp.ksize, out_chs => out_chs))
    end
    flatten_layers(layers)
end
