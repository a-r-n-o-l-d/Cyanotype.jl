m = make(EfficientNetBp(:b0, ncls=3))
@test Flux.outputsize(m, (32, 32, 3, 1)) == (3, 1)

m = make(EfficientNetBp(:b1, ncls=10))
@test Flux.outputsize(m, (32, 32, 3, 1)) == (10, 1)

m = make(EfficientNetBp(:b2, ncls=10))
@test Flux.outputsize(m, (32, 32, 3, 1)) == (10, 1)

m = make(EfficientNetBp(:b3, ncls=10))
@test Flux.outputsize(m, (32, 32, 3, 1)) == (10, 1)

m = make(EfficientNetBp(:b4, ncls=10))
@test Flux.outputsize(m, (32, 32, 3, 1)) == (10, 1)

m = make(EfficientNetBp(:b5, ncls=10))
@test Flux.outputsize(m, (32, 32, 3, 1)) == (10, 1)

m = make(EfficientNetBp(:b6, ncls=10))
@test Flux.outputsize(m, (32, 32, 3, 1)) == (10, 1)

m = make(EfficientNetBp(:b7, ncls=10))
@test Flux.outputsize(m, (32, 32, 3, 1)) == (10, 1)

m = make(EfficientNetBp(:b8, ncls=10))
@test Flux.outputsize(m, (32, 32, 3, 1)) == (10, 1)

m = make(EfficientNetBp(:small, ncls=10))
@test Flux.outputsize(m, (32, 32, 3, 1)) == (10, 1)

m = make(EfficientNetBp(:medium, ncls=10))
@test Flux.outputsize(m, (32, 32, 3, 1)) == (10, 1)

m = make(EfficientNetBp(:large, ncls=10))
@test Flux.outputsize(m, (32, 32, 3, 1)) == (10, 1)

m = make(EfficientNetBp(:xlarge, ncls=10))
@test Flux.outputsize(m, (32, 32, 3, 1)) == (10, 1)

