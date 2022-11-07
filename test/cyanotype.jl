#module CynotypeTest

using Cyanotype
using Flux
using Cyanotype: KwargsMapping, @cyanotype
#import Cyanotype: mapping

@cyanotype (
"""
Bla-bla
"""
) (
struct Foo
    a::Int = 1
    b::Int = 2
    c = 3
    d
end
)

# Checks the documentation generation
@cyanotype (
"""
Bla-bla
"""
) (
struct Foo1
    """Activation function."""
    activation
end
)
@test !isempty(eval(macroexpand(@__MODULE__, :(@doc $Foo1))))

# Checks the correctness of parametric type declaration
@cyanotype (
"""
Bla-bla
"""
) (
struct Foo2{F <: Function}
    activation::F = relu
end
)
@test Foo2().activation isa Function

# Checks the default inheritance from AbstractBlueprint
@cyanotype (
"""
Bla-bla
"""
) (
struct Foo3

end
)
@test Foo3() isa Cyanotype.AbstractBlueprint

@cyanotype (
"""
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut
labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco
laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in
voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat
cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
"""
) (
KwargsMapping(;
    flux_function = :BatchNorm,
    field_names = (:init_bias,   :init_scale, :affine, :track_stats, :epsilon, :momentum),
    flux_kwargs = (:initβ,       :initγ,      :affine, :track_stats, :ϵ,       :momentum),
    field_types = (Any,         Any,   Bool,   Bool,        :N,       :N),
    def_values  = (Flux.zeros32, Flux.ones32, true, true, 1f-5, 0.1f0)
    )
) (
struct BatchNormTest{N <: CyFloat}
    """activation function for BatchNorm layer"""
    activation
end
)

# Checks the correctness of fields declaration
for f in (:init_bias, :init_scale, :affine, :track_stats, :epsilon, :momentum)
    @test hasfield(BatchNormTest, f)
end

# Checks cyanotype function
bn = BatchNormTest(; activation = relu)
bn = cyanotype(bn; affine = false, epsilon = 0f0)
@test bn.affine == false
@test bn.epsilon == 0f0

# Checks kwargs function, for now calling of the curate function
bn = BatchNormTest(; activation = relu)
@test haskey(Cyanotype.kwargs(bn), :initβ)
@test haskey(Cyanotype.kwargs(bn), :initγ)
@test haskey(Cyanotype.kwargs(bn), :ϵ)
@test !haskey(Cyanotype.kwargs(bn), :activation)


#methods(BatchNormTest1) |> println

#println(eval(macroexpand(__module__, :(@doc BatchNormTest1)) ) )

#cya = BatchNormTest1(; activation = relu)
#cya |> println

#cyanotype(cya; affine = false) |> println
#end
