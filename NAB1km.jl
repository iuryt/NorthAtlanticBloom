using Oceananigans
using Oceananigans.Units

const sponge=Int(100kilometers)
const Nx = 96 # number of points in x
const Ny = 480 # number of points in y
const Nz = 48 # number of points in z
const H = 1000 # maximum depth


grid = RectilinearGrid(GPU(),
    size=(Nx,Ny,Nz),
    halo=(3,3,3),
    x=(-(Nx/2)kilometers, (Nx/2)kilometers), 
    y=(-(Ny/2)kilometers, (Ny/2)kilometers), 
    z=(H * cos.(LinRange(π/2,0,Nz+1)) .- H)meters,
    topology=(Bounded, Periodic, Bounded)
)

coriolis = FPlane(latitude=60)


horizontal_closure = HorizontalScalarDiffusivity(ν=1, κ=1)
vertical_closure = ScalarDiffusivity(ν=1e-4, κ=1e-4)

model = NonhydrostaticModel(grid = grid,
                            advection = UpwindBiasedFifthOrder(),
                            coriolis = coriolis,
                            closure=(horizontal_closure,vertical_closure),
                            tracers=:b, buoyancy=BuoyancyTracer())



const cz = -250meters # thermocline depth 
@inline front(x, y, z) = ((np.tanh(0.03*(z-cz))+1)/2)*(np.tanh(0.5*(x-cx))+1)/2
@inline B(x, y, z) = (front(x,z,0kilometers,cz)+front(x,z,-20kilometers,cz)+front(x,z,20kilometers,cz))/3

set!(model;b=B)