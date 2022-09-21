
module ConfigTests

using Test
using Cyanotype: AbstractCfg, @config, KwargsMapping, each_kwargs, empty_map, mappings, register_mapping!
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

@config bnmap1 struct MyBatchNormCfg1{N <: Union{Float16, Float32, Float64}}
    activation::Function = relu
end

@config "alias" bnmap2 struct MyBatchNormCfg2 # Alias is only available for concrete types in Configurations.jl
    activation::Function = relu
end

@config "alias" struct MyBatchNormCfg3
    activation::Function = relu
end

@config struct MyBatchNormCfg4
    activation::Function = relu
end

# Check the mapping registration
@test bnmap1 === mappings[:bnmap1]
@test_throws ErrorException register_mapping!(:bnmap1=>bnmap1)

# Check the correcness of fields declaration
for (k, _, _, _) ∈ each_kwargs(bnmap1)
    @test k ∈ fieldnames(MyBatchNormCfg1)
end

# Check alias construction and config accessor
@test :config ∈ fieldnames(MyBatchNormCfg2)
cfg = MyBatchNormCfg2()
@test config(cfg) isa Reflect
@test :config ∈ fieldnames(MyBatchNormCfg3)
cfg = MyBatchNormCfg3()
@test config(cfg) isa Reflect

# Check fields, accessors and default values
cfg = MyBatchNormCfg2()
@test activation(cfg) isa Function
for (field, _, T, def) ∈ each_kwargs(bnmap2)
    @test eval(:($field($cfg) == $def))
    @test eval(:($field($cfg) isa $T))
end
cfg = MyBatchNormCfg4()
@test activation(cfg) isa Function
@test activation(cfg) == relu

end # end module
