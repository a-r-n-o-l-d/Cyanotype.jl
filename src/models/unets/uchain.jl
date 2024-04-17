"""
    uchain(; encoders, decoders, bridge, connection, [paths])

Build a `Chain` with U-Net like architecture. `encoders` and `decoders` are vectors of
encoding/decoding blocks, from top to bottom (see diagram below). `bridge` is the bottom
part of U-Net  architecture. Each level of the U-Net is connected through a channel
concatenation (◊ symbol in the diagram below).

Notes :
- usually encoder unit starts with a 'MaxPool' to downsample image by 2, except the first
level encoder.
- usually decoder unit ends with a 'ConvTranspose' to upsample image by 2, except the first
level decoder.

```
┌─────────┐                                                          ┌─────────┐
│Encoder 1│                                                          │Decoder 1│
└────┬────┘                                                          └─────────┘
     │                                                                    ▴
     │                                                                    │
     ├───────────────────────────────────────────────────────────────────▸◊
     │    ┌─────────┐                                     ┌─────────┐     │
     └───▸│Encoder 2│                                     │Decoder 2├─────┘
          └────┬────┘                                     └─────────┘
               │                                               ▴
               │                                               │
               ├──────────────────────────────────────────────▸◊
               │     ┌─────────┐               ┌─────────┐     │
               └────▸│Encoder 3│               │Decoder 3├─────┘
                     └────┬────┘               └─────────┘
                          │                         ▴
                          │                         │
                          ├────────────────────────▸◊
                          │       ┌─────────┐       │
                          └──────▸│ Bridge  ├───────┘
                                  └─────────┘
```
See also [`chcat`](@ref).
"""
function uchain(; encoders, decoders, bridge, paths=fill(nothing, length(encoders)))
    length(encoders) == length(decoders) || error(
        """
        The number of encoders should be equal to the number of decoders.
        """
    )
    # build from bottom to top
    lvl = _ubridge(bridge, paths[end])
    ite = zip(reverse(encoders[2:end]), reverse(decoders[2:end]), reverse(paths[1:end - 1]))
    for (enc, dec, pth) in ite
        lvl = _uconnect(enc, lvl, dec)
        lvl = _path(lvl, pth)
    end
    _uconnect(encoders[1], lvl, decoders[1])
end

############################################################################################
#                                   INTERNAL FUNCTIONS                                     #
############################################################################################

@inline _uconnect(enc, prl, dec) = Chain(enc, prl, dec)

for T1 in [:Chain :AbstractArray], T2 in [:Chain :AbstractArray]
    @eval begin
        @inline _uconnect(enc::$T1, prl, dec::$T2) = Chain(enc..., prl, dec...)
    end
end

for T in [:Chain :AbstractArray]
    @eval begin
        @inline _uconnect(enc::$T, prl, dec) = Chain(enc..., prl, dec)
        @inline _uconnect(enc, prl, dec::$T) = Chain(enc, prl, dec...)
    end
end

@inline _ubridge(b, ::Nothing) = SkipConnection(b, chcat)

@inline _ubridge(b::AbstractArray, ::Nothing) = SkipConnection(Chain(b...), chcat)

@inline _ubridge(b, p) = Parallel(chcat, b, p)

@inline _ubridge(b::AbstractArray, p) = Parallel(chcat, Chain(b...), p)

@inline _ubridge(b, p::AbstractArray) = Parallel(chcat, b, Chain(p...))

@inline _ubridge(b::AbstractArray, p::AbstractArray) = Parallel(chcat, Chain(b...), Chain(p...))

@inline _path(lvl, ::Nothing) = SkipConnection(lvl, chcat)

@inline _path(lvl, pth) = Parallel(chcat, lvl, pth)

@inline _path(lvl, pth::AbstractArray) = Parallel(chcat, lvl, Chain(pth...))
