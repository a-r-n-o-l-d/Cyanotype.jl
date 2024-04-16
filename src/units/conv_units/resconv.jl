#ResidualBp
@cyanotype begin
    struct ResidualConvBp <: AbstractConvBp
        convolution
        connector = +
    end
end

make(bp::ResidualConvBp, ksize, channels) = SkipConnection(
    Chain(make(bp.convolution, ksize, channels)...),
    bp.connector
)
