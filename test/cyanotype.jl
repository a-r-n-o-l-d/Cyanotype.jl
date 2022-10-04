using Cyanotype
using Flux
using Cyanotype: KwargsMapping, register_mapping!, @cyanotype

bnmap1 = KwargsMapping(; # @code_warntype => OK
flux_function = :BatchNorm,
field_names = (:init_bias,   :init_scale, :affine, :track_stats, :epsilon, :momentum),
flux_kwargs = (:initβ,       :initγ,      :affine, :track_stats, :ϵ,       :momentum),
field_types = (Any,         Any,   Bool,   Bool,        :N,       :N),
def_values  = (Flux.zeros32, Flux.ones32, true, true, 1f-5, 0.1f0)
);

register_mapping!(:bnmap1=>bnmap1)

"""
pouet pouet
"""
BatchNormTest1

@cyanotype bnmap1 struct BatchNormTest1{N <: CyFloat} <: Cyanotype.AbstractCyanotype
    """activation function for BatchNorm layer"""
    activation
end

methods(BatchNormTest1) |> println

println(eval(macroexpand(Main, :(@doc BatchNormTest1)) ) )

cya = BatchNormTest1(; activation = relu)
cya |> println

cyanotype(cya; affine = false) |> println
