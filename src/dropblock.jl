# package DropBlockLayer

#=

# https://github.com/FluxML/Metalhead.jl/blob/master/src/layers/drop.jl
# https://github.com/pytorch/vision/blob/main/torchvision/ops/drop_block.py
# https://github.com/clguo/SA-UNet/blob/master/Dropblock.py => sync channels

"""
    DropBlock(proba = 0.95, bsize = 7, rng = rng_from_array())  gscale = 1.0

The `DropBlock` layer. While training, it zeroes out continguous regions of size
`block_size` in the input. During inference, it simply returns the input `x`.It can be used
in two ways: either with all blocks having the same survival probability or with a linear
scaling rule across the blocks. This is performed only at training time. At test time, the
`DropBlock` layer is equivalent to `identity`.
([reference](https://arxiv.org/abs/1810.12890))

# Arguments
  - `proba`: probability of keeping a block. If `nothing` is passed, it returns
    `identity`.
  - `bsize`: size of the block to drop
  - `gscale`: multiplicative factor for `gamma` used. For the calculation of gamma, refer
    to [the paper](https://arxiv.org/abs/1810.12890).
  - `rng`: can be used to pass in a custom RNG instead of the default. Custom RNGs are only
    supported on the CPU.
"""
# gscale => pscale, 0 < pscale ≤ 1
# il faut une fonction schedule_dropblock!(layers::Chain) schedropblock!
# qui pour chaque DropBlock dans layers change proba = proba * pscale
mutable struct DropBlock{F,R<:AbstractRNG}
    proba::F #pkeep
    bsize::Integer
    #gscale::F # for scheduled dropblock ?
    active::Union{Bool, Nothing}
    rng::R
end

@functor DropBlock

trainable(::DropBlock) = (;)

function DropBlock(proba = 0.95, bsize = 7, rng = rng_from_array()) #::Integer, , gscale = 1.0
    if isnothing(proba)
        identity
    else
        DropBlock(proba, bsize, nothing, rng) #, gscale
    end
end

function Base.show(io::IO, d::DropBlock)
    print(io, "DropBlock(", d.proba)
    print(io, ", bsize = $(repr(d.bsize))")
    #print(io, ", gscale = $(repr(d.gscale))")
    print(io, ")")
end

function (db::DropBlock)(x)
    _dropblock_checks(db, x)
    Flux._isactive(db) ? _dropblock(db, x) : x
end

function Flux.testmode!(db::DropBlock, mode = true)
    (db.active = (isnothing(mode) || mode === :auto) ? nothing : !mode; db)
end

function _dropblock_checks(db::DropBlock, x::AbstractArray{<:Any, N}) where N #(rng::AbstractRNG, x::AbstractArray{<:Any, N}, proba, gscale)
    4 ≤ N ≤ 5 || error("x must be an array with 4 or 5 dimensions for DropBlock.")
    0 ≤ db.proba ≤ 1 || error("proba must be between 0 and 1, got $db.proba") # ]0; 1] ?
    #0 ≤ db.gscale ≤ 1 || error("gscale must be between 0 and 1, got $db.gscale")
    _dropblock_checks(db.rng, x) || error("""x is a CuArray, but rng isa $(typeof(rng)).
        DropBlock only supports CUDA.RNG for CuArrays.""")
end

ChainRulesCore.@non_differentiable _dropblock_checks(db, x)

_dropblock_checks(::CUDA.RNG, ::CuArray) = true

_dropblock_checks(::Any, ::Any) = true

_dropblock_checks(rng::AbstractRNG, ::CuArray) = false

function _dropblock(db::DropBlock, x::AbstractArray{T, 4}) where T
    h, w, _, _ = size(x)
    n = h * w
    sz = min(db.bsize, h, w)
    gamma =  (1 - db.proba) * n / (sz^2 * (w - sz + 1) * (h - sz + 1))
    #gamma *= db.gscale
    mask = rand_like(db.rng, x)
    mask .= mask .< gamma
    mask = 1 .- maxpool(mask, (sz, sz); stride = 1, pad = sz ÷ 2)
    norm = length(mask) / (sum(mask) .+ eps(T)) # T(1e-6)
    x .* mask .* norm
end

function _dropblock(db::DropBlock, x::AbstractArray{T, 5}) where T
    h, w, d, _, _ = size(x)
    n = h * w * d
    sz = min(db.bsize, h, w, d)
    gamma = (1 - db.proba) * n / (sz^3 * (w - sz + 1) * (h - sz + 1) * (d - sz + 1))
    #gamma *= db.gscale
    mask = rand_like(db.rng, x)
    mask .= mask .< gamma
    mask = 1 .- maxpool(mask, (sz, sz, sz); stride = 1, pad = sz ÷ 2)
    norm = length(mask) / (sum(mask) .+ eps(T)) # T(1e-6)
    x .* mask .* norm
end

ChainRulesCore.@non_differentiable _dropblock(db, x)


function _dropblock_checks(x::AbstractArray{<:Any, 4}, drop_block_prob, gamma_scale)
    @assert 0 ≤ drop_block_prob ≤ 1 "drop_block_prob must be between 0 and 1, got $drop_block_prob"
    @assert 0 ≤ gamma_scale ≤ 1 "gamma_scale must be between 0 and 1, got $gamma_scale"
end

function _dropblock_checks(x, drop_block_prob, gamma_scale)
    throw(ArgumentError("x must be an array with 4 dimensions (H, W, C, N) for DropBlock."))
end

ChainRulesCore.@non_differentiable _dropblock_checks(x, drop_block_prob, gamma_scale)

# TODO add experimental `DropBlock` options from timm such as gaussian noise and
# more precise `DropBlock` to deal with edges (#188)
"""
    dropblock([rng = rng_from_array(x)], x::AbstractArray{T, 4}, drop_block_prob, block_size,
              gamma_scale, active::Bool = true)
The dropblock function. If `active` is `true`, for each input, it zeroes out continguous
regions of size `block_size` in the input. Otherwise, it simply returns the input `x`.
# Arguments
  - `rng`: can be used to pass in a custom RNG instead of the default. Custom RNGs are only
    supported on the CPU.
  - `x`: input array
  - `drop_block_prob`: probability of dropping a block. If `nothing` is passed, it returns
    `identity`.
  - `block_size`: size of the block to drop
  - `gamma_scale`: multiplicative factor for `gamma` used. For the calculations,
    refer to [the paper](https://arxiv.org/abs/1810.12890).
If you are not a package developer, you most likely do not want this function. Use [`DropBlock`](@ref)
instead.
"""
function dropblock(rng::AbstractRNG, x::AbstractArray{T, 4}, drop_block_prob,
                   block_size::Integer, gamma_scale) where {T}
    H, W, _, _ = size(x)
    total_size = H * W
    clipped_block_size = min(block_size, min(H, W))
    gamma = gamma_scale * drop_block_prob * total_size / clipped_block_size^2 /
            ((W - block_size + 1) * (H - block_size + 1))
    block_mask = dropblock_mask(rng, x, gamma, clipped_block_size)
    normalize_scale = length(block_mask) / sum(block_mask) .+ T(1e-6)
    return x .* block_mask .* normalize_scale
end

## bs is `clipped_block_size`
# Dispatch for GPU
dropblock_mask(rng::CUDA.RNG, x::CuArray, gamma, bs) = _dropblock_mask(rng, x, gamma, bs)

function dropblock_mask(rng, x::CuArray, gamma, bs)
    throw(ArgumentError("x isa CuArray, but rng isa $(typeof(rng)). dropblock only supports
                        CUDA.RNG for CuArrays."))
end

# Dispatch for CPU
dropblock_mask(rng, x, gamma, bs) = _dropblock_mask(rng, x, gamma, bs)

# Generates the mask to be used for `DropBlock`
@inline function _dropblock_mask(rng, x::AbstractArray{T, 4}, gamma,
                                 clipped_block_size::Integer) where {T}
    block_mask = rand_like(rng, x)
    block_mask .= block_mask .< gamma
    return 1 .- maxpool(block_mask, (clipped_block_size, clipped_block_size);
                   stride = 1, pad = clipped_block_size ÷ 2)
end

ChainRulesCore.@non_differentiable _dropblock_mask(rng, x, gamma, clipped_block_size)
=#
