module CynotypeTest

using Cyanotype
using Flux
using Cyanotype: KwargsMapping, register_mapping!, @cyanotype
#import Cyanotype: mapping

bnmap1 = KwargsMapping(; # @code_warntype => OK
flux_function = :BatchNorm,
field_names = (:init_bias,   :init_scale, :affine, :track_stats, :epsilon, :momentum),
flux_kwargs = (:initβ,       :initγ,      :affine, :track_stats, :ϵ,       :momentum),
field_types = (Any,         Any,   Bool,   Bool,        :N,       :N),
def_values  = (Flux.zeros32, Flux.ones32, true, true, 1f-5, 0.1f0)
);

register_mapping!(:bnmap1=>bnmap1)

@cyanotype bnmap1 (
"""
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut
labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco
laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in
voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat
cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
"""
) (
struct BatchNormTest1{N <: CyFloat} <: Cyanotype.AbstractBlueprint
    """activation function for BatchNorm layer"""
    activation
end
)

#methods(BatchNormTest1) |> println

#println(eval(macroexpand(__module__, :(@doc BatchNormTest1)) ) )

#cya = BatchNormTest1(; activation = relu)
#cya |> println

#cyanotype(cya; affine = false) |> println
end
