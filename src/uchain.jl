"""
    uchain(;encoders, decoders, bridge, connection)

Build a `Chain` with U-Net like architecture. `encoders` and `decoders` are vectors of
encoding/decoding blocks, from top to bottom (see diagram below). `bridge` is the bottom
part of U-Net  architecture. Each level of the U-Net is connected through a channel
concatenation (◊ symbol in the diagram below).

Notes :
- usually encoder block starts with a 'MaxPool' to downsample image by 2, except
the first level encoder.
- usually decoder block ends with a 'ConvTranspose' to upsample image by 2,
except the first level decoder.

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
function uchain(; encoders, decoders, bridge)
    length(encoders) == length(decoders) || error("""
    The number of encoders should be equal to the number of decoders.
    """)
    # build from bottom to top
    l = _ubridge(bridge)
    for (e, d) in zip(reverse(encoders[2:end]), reverse(decoders[2:end]))
        l = _uconnect(e, l, d)
        l = SkipConnection(l, chcat) # Parallel(chcat, l, cross_connector_block)
    end
    _uconnect(encoders[1], l, decoders[1])
end

############################################################################################
#                                   INTERNAL FUNCTIONS                                     #
############################################################################################

@inline _uconnect(enc, prl, dec) = Chain(enc, prl, dec)

for T1 in [:Chain :AbstractArray], T2 in [:Chain :AbstractArray]
    @eval begin
        @inline _uconnect(enc::($T1), prl, dec::($T2)) = Chain(enc..., prl, dec...)
    end
end

for T in [:Chain :AbstractArray]
    @eval begin
        @inline _uconnect(enc::($T), prl, dec) = Chain(enc..., prl, dec)
        @inline _uconnect(enc, prl, dec::($T)) = Chain(enc, prl, dec...)
    end
end

@inline _ubridge(b) = SkipConnection(b, chcat)

@inline _ubridge(b::AbstractArray) = SkipConnection(Chain(b...), chcat)
