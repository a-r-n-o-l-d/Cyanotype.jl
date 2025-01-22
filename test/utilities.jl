hac = HybridAtrouConvBp()
hac2 = spread(hac; act=leakyrelu)
@test hac2.conv.norm.act == leakyrelu

hac2 = spread(hac2, :act, leakyrelu => relu)
@test hac2.conv.norm.act == relu

bp = FusedMbConvBp(stride=2, exch=6, init=Flux.glorot_normal)
conv = spread(bp; stride = 1, skip=true)
bp.conv.stride