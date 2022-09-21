using Cyanotype: KwargsMapping, mappings, register_mapping!, empty_map

@testset "KwargsMapping" begin
    # Test mapping for Flux.BatchNorm kwargs
    bnmap = KwargsMapping(;
        field_names    = (:init_bias,   :init_scale, :affine, :track_stats, :epsilon, :momentum),
        flux_names     = (:initβ,       :initγ,      :affine, :track_stats, :ϵ,       :momentum),
        field_types    = (:Function,    :Function,   :Bool,   :Bool,        :N,       :N),
        field_defaults = (Flux.zeros32, Flux.ones32, true,    true,         1f-5,     0.1f0) )

    register_mapping!(:bnmap=>bnmap)

    @test bnmap === mappings[:bnmap]

    @test_throws ErrorException register_mapping!(:empty_map=>empty_map)
end
