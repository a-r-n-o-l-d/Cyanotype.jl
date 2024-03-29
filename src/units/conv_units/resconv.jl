#ResidualBp
@cyanotype begin
    struct ResidualConvBp{C<:AbstractConvBp,F<:Function} <: AbstractConvBp
        convolution::C
        connector::F = +
    end
end

make(bp::ResidualConvBp, ksize, channels) = SkipConnection(
    Chain(make(bp.convolution, ksize, channels)...),
    bp.connector
)
