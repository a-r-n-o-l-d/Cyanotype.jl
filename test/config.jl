
module ConfigTests

using Test
using Cyanotype
using Cyanotype: AbstractCyano, @config, KwargsMapping, each_kwargs, empty_map, mappings, register_mapping!, currate_kwargs
using Flux: relu, zeros32, ones32
using Configurations: Reflect

# Mapping for Flux.BatchNorm kwargs
bnmap1 = KwargsMapping(;
    field_names    = (:init_bias,   :init_scale, :affine, :track_stats, :epsilon, :momentum),
    flux_names     = (:initβ,       :initγ,      :affine, :track_stats, :ϵ,       :momentum),
    field_types    = (:Function,    :Function,   :Bool,   :Bool,        :N,       :N),
    field_defaults = (zeros32,      ones32, true,    true,         1f-5,     0.1f0) )
register_mapping!(:bnmap1=>bnmap1)

FFloat = Union{Float16, Float32, Float64}
bnmap2 = KwargsMapping(;
    field_names    = (:init_bias,   :init_scale, :affine, :track_stats, :epsilon, :momentum),
    flux_names     = (:initβ,       :initγ,      :affine, :track_stats, :ϵ,       :momentum),
    field_types    = (:Function,    :Function,   :Bool,   :Bool,        :FFloat,  :FFloat),
    field_defaults = (zeros32,      ones32, true,    true,         1f-5,     0.1f0) )
register_mapping!(:bnmap2=>bnmap2)

@config bnmap1 struct BatchNormCyano1{N <: Union{Float16, Float32, Float64}} <: AbstractCyano
    activation::Function = relu
end

# Alias is only available for concrete types in Configurations.jl
@config "alias" bnmap2 struct BatchNormCyano2 <: AbstractCyano
    activation::Function = relu
end

@config "alias" struct BatchNormCyano3 <: AbstractCyano
    activation::Function = relu
end

@config struct BatchNormCyano4 <: AbstractCyano
    activation::Function = relu
end

# Check the mapping registration
@test bnmap1 === mappings[:bnmap1]
@test_throws ErrorException register_mapping!(:bnmap1=>bnmap1)

# Check the correcness of fields declaration
for (k, _, _, _) ∈ each_kwargs(bnmap1)
    @test k ∈ fieldnames(BatchNormCyano1)
end

# Check alias construction and config accessor
@test :config ∈ fieldnames(BatchNormCyano2)
cfg = BatchNormCyano2()
@test config(cfg) isa Reflect
@test :config ∈ fieldnames(BatchNormCyano3)
cfg = BatchNormCyano3()
@test config(cfg) isa Reflect

# Check fields, accessors and default values
cfg = BatchNormCyano2()
@test activation(cfg) isa Function
for (field, _, T, def) ∈ each_kwargs(bnmap2)
    @test eval(:($field($cfg) == $def))
    @test eval(:($field($cfg) isa $T))
end
cfg = BatchNormCyano4()
@test activation(cfg) isa Function
@test activation(cfg) == relu

# Check currate_kwargs function
cfg = BatchNormCyano1()
@test mapping(cfg) == bnmap1
@test haskey(currate_kwargs(cfg, mapping(cfg)), :initβ)
@test haskey(currate_kwargs(cfg, mapping(cfg)), :initγ)
@test haskey(currate_kwargs(cfg, mapping(cfg)), :ϵ)
@test !haskey(currate_kwargs(cfg, mapping(cfg)), :activation)

# Check copy constructor generation
cfg = BatchNormCyano1(cfg; affine = true)
@test affine(cfg) == true
@test hasmethod(BatchNormCyano1, (BatchNormCyano1,), (:activation, :affine))

end # end module
