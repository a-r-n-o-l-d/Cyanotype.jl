abstract type AbstractBpConv <: AbstractBlueprint end

const CyPad = Union{SamePad,Int}

@cyanotype begin
    KwargsMapping(
            flfunc = :Conv,
            fnames = (:stride, :pad,      :dilation, :groups, :init), #stride, bias
            flargs = (:stride, :pad,      :dilation, :groups, :init),
            ftypes = (Int,     :P,        Int,       Int,     :I),
            defval = (1,       SamePad(), 1,         1,       glorot_uniform)
        )

    """
    BpConv(; kwargs...)

    A cyanotype blueprint describing a convolutionnal module or layer depending om the value
    of `normalization` argument.
    """
    struct BpConv{N<:Union{Nothing,AbstractBpNorm},A,I<:Function,P<:CyPad} <: AbstractBpConv
        @volume
        @activation(identity)
        """
        `normalization`:
        """
        normalization::N = nothing # BpNoNorm()
        """
        `depthwise`:
        """
        depthwise::Bool = false
        # residual
        """
        `revnorm`:
        """
        revnorm::Bool = false
        """
        `preactivation`:
        """
        preactivation::Bool = false
        """
        `bias`:
        """
        bias::Bool = normalization isa Nothing
    end
end

#=function make(bp::BpConv; ksize=3, channels)
    k = genk(ksize, bp.volume)
    #_build_conv(bp.normalization, bp, k, channels) #|> flatten_layers
    _make_conv(bp, k, channels) #|> flatten_layers
end=#

make(bp::BpConv, ksize, channels::Int) = make(bp, ksize, channels => channels)

# Regular convolutionnal layer
function make(bp::BpConv{<:Nothing}, ksize, channels::Pair) #where {N<:Nothing}
    k = genk(ksize, bp.volume)
    kw = kwargs(bp)
    if bp.depthwise
        kw[:groups] = first(channels)
    end
    Conv(k, channels, bp.activation; kw...) #|> flatten_layers
end

# Convolutionnal unit: convolutionnal layer & normalization layer
function make(bp::BpConv{<:AbstractBpNorm}, ksize, channels::Pair) #where {N<:AbstractBpNorm}
    k = genk(ksize, bp.volume)
    layers = []
    in_chs, out_chs = channels
    activation = bp.normalization.activation
    kw = kwargs(bp)
    if bp.depthwise
        kw[:groups] = in_chs
    end
    # Normalization first
    if bp.revnorm
        # Activation before convolution ?
        if bp.preactivation
            act_n = activation
            act_c = identity
        else
            act_n = identity
            act_c = activation
        end
        norm = cyanotype(bp.normalization; activation = act_n)
        conv = Conv(k, channels, act_c; bias = bp.bias, kw...)
        push!(layers, make(norm, in_chs), conv)
    # Convolution first
    else
        # Activation before convolution ?
        if bp.preactivation
            act_n = identity
            push!(layers, activation)
        else
            act_n = activation
        end
        norm = cyanotype(bp.normalization; activation = act_n)
        conv = Conv(k, channels; bias = bp.bias, kw...)
        push!(layers, conv, make(norm, out_chs))
    end
    flatten_layers(layers)
end

@cyanotype begin
    """
    BpDConv(; kwargs)

    Describes a convolutionnal module formed by two successive convolutionnal modules.
    """
    struct BpDConv{C1<:AbstractBpConv,
                             C2<:AbstractBpConv} <: AbstractBpConv
        @volume #enlever
        conv1::C1 #conv1
        conv2::C2 = conv1 #conv2
    end
end

# channels::Pair in_chs=>out_chs out_chs=>out_chs
# channels::NTuple{3} in_chs=>mid_chs mid_chs=>out_chs
function make(bp::BpDConv, ksize, channels)
    # convolution1.vol == convolution2.vol || error("")
    c1 = spread(bp.conv1; vol = bp.volume) #cyanotype(bp.convolution1; vol = bp.volume)
    c2 = spread(bp.conv2; vol = bp.volume) #cyanotype(bp.convolution2; vol = bp.volume)
    in_chs, mid_chs, out_chs = channels
    [
        make(c1, ksize, in_chs=>mid_chs),
        make(c2, ksize, mid_chs=>out_chs)
    ] |> flatten_layers
end

# Peut-etre inutile
@cyanotype begin
    """
    Template describing a module with N `NConvBp` repeated.
    """
    struct BpNConv{C<:AbstractBpConv} <: AbstractBpConv
        convolution::C
        nrepeat::Int
    end
end

function make(bp::BpNConv, ksize, channels)
    layers = []
    in_chs, out_chs = channels
    for _ in 1:bp.nrepeat
        push!(layers, make(bp.convolution, ksize, in_chs=>out_chs))
        in_chs = out_chs
    end
    flatten_layers(layers)
end

@cyanotype begin
    """
    aka Hybrid Dilated Convolution
    [paper](@ref https://doi.org/10.1109/WACV.2018.00163)
    [example](@ref https://doi.org/10.1016/j.image.2019.115664)
    [example](@ref https://doi-org/10.1109/ICMA54519.2022.9855903)
    """
    struct BpHybridAtrouConv{N,C<:BpConv} <: AbstractBpConv #BpHybridAtrouConv
        dilation_rates::NTuple{N,Int} = (1, 2, 3)
        conv::C = BpConv(normalization=BpBatchNorm())
    end
end

function make(bp::BpHybridAtrouConv, ksize, channels)
    _check_dilation_rates(ksize, bp.dilation_rates) || error("Invalid dilation rates.")
    layers = []
    in_chs, out_chs = channels
    for dr in bp.dilation_rates
        c = cyanotype(bp.conv; dilation = dr)
        push!(layers, make(c, ksize, in_chs=>out_chs)...)
        in_chs = out_chs
    end
    flatten_layers(layers)
end

@cyanotype constructor=false begin
    """

    """
    struct BpPointwiseConv <: AbstractBpConv
        conv::BpConv
    end
end

#BpPointwiseConv() = BpPointwiseConv(BpConv())

BpPointwiseConv(; kwargs...) = BpPointwiseConv(BpConv(; kwargs...))

make(bp::BpPointwiseConv, channels) = make(bp.conv, 1, channels)

@cyanotype constructor=false begin
    """

    """
    struct BpChannelExpansion <: AbstractBpConv
        expansion::Int
        conv::BpPointwiseConv
    end
end

function BpChannelExpansion(; kwargs...)
    kw = Dict(kwargs)
    expansion = kw[:expansion]
    delete!(kw, :expansion)
    conv =  BpPointwiseConv(; kw...)
    BpChannelExpansion(expansion, conv)
end

function make(bp::BpChannelExpansion, channels)
    if bp.expansion <= 1
        identity
    else
        make(bp.conv, channels => channels * bp.expansion)
    end
end

@cyanotype constructor=false begin
    """

    """
    struct BpDepthwiseConv <: AbstractBpConv
        conv::BpConv
    end
end

function BpDepthwiseConv(; kwargs...)
    if haskey(kwargs, :depthwise)
        kwargs[:depthwise] = true
    end
    BpDepthwiseConv(BpConv(; kwargs...))
end

make(bp::BpDepthwiseConv, ksize, channels) = make(bp.conv, ksize, channels)

#=
function make(bp::BpPointwiseConv{<:Nothing}, channels)
    k = genk(1, bp.volume)
    Conv(k, channels, bias=true)
end

function make(bp::BpPointwiseConv{<:AbstractBpNorm}, channels)
    k = genk(1, bp.volume)
    [
        Conv(k, channels, bias=false),
        make(bp.norm, last(channels))
    ] |> flatten_layers
end
=#

############################################################################################
#                                   INTERNAL FUNCTIONS                                     #
############################################################################################
#=
# Regular convolutionnal layer
function _make_conv(bp::BpConv{N}, k, chs) where {N<:Nothing}
    kw = kwargs(bp)
    if bp.depthwise
        kw[:groups] = first(chs)
    end
    [Conv(k, chs, bp.activation; kw...)]
end

# Convolutionnal unit: convolutionnal layer & normalization layer
function _make_conv(bp::BpConv{N}, k, chs) where {N<:AbstractBpNorm}
    layers = []
    in_chs, out_chs = chs
    activation = bp.normalization.activation
    kw = kwargs(bp)
    if bp.depthwise
        kw[:groups] = first(chs)
    end
    # Normalization first
    if bp.revnorm
        # Activation before convolution ?
        if bp.preactivation
            act_n = activation
            act_c = identity
        else
            act_n = identity
            act_c = activation
        end
        norm = cyanotype(bp.normalization; activation = act_n)
        conv = Conv(k, chs, act_c; bias = bp.bias, kw...)
        push!(layers, make(norm; channels = in_chs), conv)
    # Convolution first
    else
        # Activation before convolution ?
        if bp.preactivation
            act_n = identity
            push!(layers, activation)
        else
            act_n = activation
        end
        norm = cyanotype(bp.normalization; activation = act_n)
        conv = Conv(k, chs; bias = bp.bias, kw...)
        push!(layers, conv, make(norm; channels = out_chs))
    end
    flatten_layers(layers)
end
=#
#=
# A usual convolutionnal layer
function _build_conv(::BpNoNorm, bp, k, chs)
    [Conv(k, chs, bp.normalization.activation; kwargs(bp)...)]
end

# Convolutionnal module: convolutionnal layer & normalization layer
function _build_conv(nm, bp, k, chs)
    layers = []
    in_chs, out_chs = chs
    activation = bp.normalization.activation
    # Normalization first
    if bp.revnorm
        # Activation before convolution ?
        if bp.preactivation
            act_n = activation
            act_c = identity
        else
            act_n = identity
            act_c = activation
        end
        norm = cyanotype(nm; activation = act_n)
        conv = Conv(k, chs, act_c; bias = bp.bias, kwargs(bp)...)
        push!(layers, make(norm; channels = in_chs), conv)
    # Convolution first
    else
        # Activation before convolution ?
        if bp.preactivation
            act_n = identity
            push!(layers, activation)
        else
            act_n = activation
        end
        norm = cyanotype(nm; activation = act_n)
        conv = Conv(k, chs; bias = bp.bias, kwargs(bp)...)
        push!(layers, conv, make(norm; channels = out_chs))
    end
    flatten_layers(layers)
end
=#

#https://arxiv.org/pdf/1702.08502.pdf
# DOI 10.1109/WACV.2018.00163
function _check_dilation_rates(k, dr)
    issorted(dr) || error("Dilation rates must be increasing.")
    M = dr[end]
    i = length(dr)
    while i > 2
        i = i - 1
        M = max(M - 2 * dr[i], M - 2 * (M - dr[i]), dr[i])
    end
    M <= k
end
