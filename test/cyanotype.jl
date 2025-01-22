# ToDo: check correcness without constructor

@cyanotype begin
    """
    Bla-bla
    """
    struct Foo
        a::Int = 1
        b::Int
        c = 3
        d
        Cyanotype.@volume
    end
end

# Check the correctness of kwargs constructor
f = Foo(; a=42, b=84, c=666, d=1.618, vol=false)
@test f.a == 42

# Check the documentation generation
@cyanotype begin
    """
    Bla-bla
    """
    struct Foo1
        """Activation function."""
        act
    end
end
println(eval(macroexpand(@__MODULE__, :(@doc $Foo1))))
#@test !isempty(eval(macroexpand(@__MODULE__, :(@doc $Foo1))))

# Check the correctness of parametric type declaration
@cyanotype begin
    """
    Bla-bla
    """
    struct Foo2{F <: Function}
        act::F = relu
    end
end
@test Foo2().act isa Function

# Check the default inheritance from AbstractBlueprint
@cyanotype begin
    """
    Bla-bla
    """
    struct Foo3

    end
end
@test Foo3() isa Cyanotype.AbstractBlueprint

@cyanotype begin
    KwargsMapping(
        flfunc = :BatchNorm,
        fnames = (:init_bias, :init_scale, :affine, :track_stats, :epsilon, :momentum),
        flargs = (:initβ,     :initγ,      :affine, :track_stats, :ϵ,       :momentum),
        defval = (Flux.zeros32, Flux.ones32, true, true, 1f-5, 0.1f0)
    )

    """
    Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor
    incididunt.
    """
    struct BatchNormTest
        """activation function for BatchNorm layer"""
        act
    end
end

# Check the correctness of fields declaration
for f in (:init_bias, :init_scale, :affine, :track_stats, :epsilon, :momentum)
    @test hasfield(BatchNormTest, f)
end

# Check cyanotype function
bn = BatchNormTest(; act = relu)
bn = cyanotype(bn; affine = false, epsilon = 0f0)
@test bn.affine == false
@test bn.epsilon == 0f0

# Check kwargs function
bn = BatchNormTest(; act = relu)
@test haskey(Cyanotype.kwargs(bn), :initβ)
@test haskey(Cyanotype.kwargs(bn), :initγ)
@test haskey(Cyanotype.kwargs(bn), :ϵ)
@test !haskey(Cyanotype.kwargs(bn), :act)
