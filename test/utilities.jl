hac = HybridAtrouConvBp()
hac2 = spread(hac; act=leakyrelu)
@test hac2.conv.norm.act == leakyrelu
