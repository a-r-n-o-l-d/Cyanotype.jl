#ResidualBp
@cyanotype begin
    struct ResidualConvBp <: AbstractConvBp
        conv
        connector = +
    end
end

make(bp::ResidualConvBp, ksize, channels) = SkipConnection(
    Chain(make(bp.conv, ksize, channels)...),
    bp.connector
)
