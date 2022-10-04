using Cyanotype: KwargsMapping, register_mapping!
using Flux: zeros32, ones32, relu
using Cyanotype: @cyano, AbstractCyanotype, eachkwargs, curate

# Mapping for Flux.BatchNorm kwargs
bnmap1 = KwargsMapping(;
    flux_function = :BatchNorm,
    field_names    = (:init_bias,   :init_scale, :affine, :track_stats, :epsilon, :momentum),
    flux_names     = (:initβ,       :initγ,      :affine, :track_stats, :ϵ,       :momentum),
    field_types    = (:Any,    :Any,   :Bool,   :Bool,        :N,       :N),
    field_defaults = (zeros32,      ones32, true,    true,         1f-5,     0.1f0)
)
try
    register_mapping!(:bnmap1=>bnmap1)
catch

end

#=
bnmap2 = KwargsMapping(;
    flux_function = :BatchNorm,
    field_names    = (:init_bias,   :init_scale, :affine, :track_stats, :epsilon, :momentum),
    flux_names     = (:initβ,       :initγ,      :affine, :track_stats, :ϵ,       :momentum),
    field_types    = (:Function,    :Function,   :Bool,   :Bool,        :AbstractFloat,  :AbstractFloat),
    field_defaults = (zeros32,      ones32, true,    true,         1f-5,     0.1f0)
)
register_mapping!(:bnmap2=>bnmap2)
=#
function test_mapping(kmap1::T, kmap2::T) where T <: KwargsMapping
    @test kmap1.flux_function === kmap2.flux_function
    @test kmap1.field_names === kmap2.field_names
    @test kmap1.field_types === kmap2.field_types
    @test kmap1.field_defaults === kmap2.field_defaults
end

function test_mapping(kmap1::KwargsMapping, kmap2::Symbol)
    test_mapping(kmap1, Cyanotype.MAPPINGS[kmap2])
end

# Checks the mapping registration
test_mapping(bnmap1, :bnmap1)
#@test_throws ErrorException register_mapping!(:bnmap1=>bnmap1)

# Checks the documentation generation
@cyano struct EmptyTest <: AbstractCyanotype
    """Activation function."""
    activation
end
@test !isempty(eval(macroexpand(@__MODULE__, :(@doc $EmptyTest))))

# Checks the correctness of parametric type declaration
@cyano struct EmptyTest2{F <: Function}
    activation::F = relu
end
@test EmptyTest2().activation isa Function

# Checks the default inheritance from AbstractCyanotype
@cyano struct EmptyTest3
    activation = relu
end
@test EmptyTest3() isa AbstractCyanotype

# Checks the declaration and construction of a tagging struct
@cyano struct EmptyTest4 end
@test EmptyTest4() isa AbstractCyanotype

# A more complete use case: wraps Flux.BatchNorm
@cyano bnmap1 struct BatchNormTest{N <: Union{Float16, Float32, Float64}} <: AbstractCyanotype
    """activation function for BatchNorm layer"""
    activation = relu
end

# Checks the correctness of fields declaration
for (k, _, _, _) ∈ eachkwargs(bnmap1)
    @test k ∈ fieldnames(BatchNormTest)
end

# Checks generated functions
cfg = BatchNormTest(; epsilon = 1.0, momentum = 2.0)
@test all(keys(getfields(cfg)) .== fieldnames(BatchNormTest))
@test all(values(getfields(cfg)) .== [getfield(cfg, f) for f in fieldnames(BatchNormTest)])
test_mapping(mapping(cfg), :bnmap1)
test_mapping(mapping(BatchNormTest), :bnmap1)

# Checks currate function, for now calling of the curate function need to know the calling
# module if outside the Cynanotype module.
cfg = BatchNormTest()
@test haskey(curate(cfg, @__MODULE__), :initβ)
@test haskey(curate(cfg, @__MODULE__), :initγ)
@test haskey(curate(cfg, @__MODULE__), :ϵ)
@test !haskey(curate(cfg, @__MODULE__), :activation)

# Checks copy constructor generation
cfg = BatchNormTest(cfg; affine = false, epsilon = 0f0)
@test cfg.affine == false
@test cfg.epsilon == 0f0



@cyano struct EmptyTest5 <: AbstractCyanotype
    Cyanotype.@activation(relu)
    Cyanotype.@volumetric
end

EmptyTest5() |> println
