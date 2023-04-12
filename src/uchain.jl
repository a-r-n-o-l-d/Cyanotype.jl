"""
    uchain(;encoders, decoders, bridge, connection)

Build a `Chain` with U-Net like architecture. `encoders` and `decoders` are
vectors of encoding/decoding blocks, from top to bottom (see diagram below).
`bridge` is the bottom part of U-Net  architecture.
Each level of the U-Net is connected through a 2-argument callable `connection`.
`connection` could be a vector in (unusual) case the way levels are connected
vary from one level to another.

Notes :
- usually encoder block starts with a 'MaxPool' to downsample image by 2, except
the first level encoder.
- usually decoder block ends with a 'ConvTranspose' to upsample image by 2,
except the first level decoder.

```
+---------+                                                          +---------+
|encoder 1|                                                          |decoder 1|
+---------+                                                          +---------+
     |------------------------------------------------------------------->^
     |                                                                    |
     |   +---------+                                         +---------+  |
     +-->|encoder 2|                                         |decoder 2|--+
         +---------+                                         +---------+
              |-------------------------------------------------->^
              |                                                   |
              |   +---------+                        +---------+  |
              +-->|encoder 3|                        |decoder 3|--+
                  +---------+                        +---------+
                       |--------------------------------->^
                       |                                  |
                       |   +---------+       +---------+  |
                       +-->|encoder 4|       |decoder 4|--+
                           +---------+       +---------+
                                |---------------->^
                                |                 |
                                |    +--------+   |
                                +--->| bridge |---+
                                     +--------+
```
See also [`chcat`](@ref).
"""
function uchain(; encoders, decoders, bridge)
    length(encoders) == length(decoders) || error("""
    The number of encoders should be equal to the number of decoders.
    """)
    # build from bottom to top
    l = ubridge(bridge)
    for (e, d) in zip(reverse(encoders[2:end]), reverse(decoders[2:end]))
        l = uconnect(e, l, d)
        l = SkipConnection(l, chcat) # Parallel(c, l, cross_connector_block)
    end
    uconnect(encoders[1], l, decoders[1])
end

@inline uconnect(enc, prl, dec) = Chain(enc, prl, dec)

for T1 in [:Chain :AbstractArray], T2 in [:Chain :AbstractArray]
    @eval begin
        @inline uconnect(enc::($T1), prl, dec::($T2)) = Chain(enc..., prl, dec...)
    end
end

for T in [:Chain :AbstractArray]
    @eval begin
        @inline uconnect(enc::($T), prl, dec) = Chain(enc..., prl, dec)
        @inline uconnect(enc, prl, dec::($T)) = Chain(enc, prl, dec...)
    end
end

@inline ubridge(b) = SkipConnection(b, chcat)

@inline ubridge(b::AbstractArray) = SkipConnection(Chain(b...), chcat)
