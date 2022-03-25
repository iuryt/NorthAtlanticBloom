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
vertical_closure = ScalarDiffusivity(ν=1e-3, κ=1e-3)

model = NonhydrostaticModel(grid = grid,
                            advection = UpwindBiasedFifthOrder(),
                            coriolis = coriolis,
                            closure=(horizontal_closure,vertical_closure),
                            tracers=:b, buoyancy=BuoyancyTracer())



const cz = -250meters # thermocline depth 
const L = 50kilometers
const amp = 1kilometers
@inline front(x, y, z, cx) = ((tanh(0.03*(z-cz))+1)/2)*(tanh(0.5*(x-(cx+sin(2pi*y/L)*amp)))+1)/2
@inline B(x, y, z) = (front(x,y,z,0kilometers)+front(x,y,z,-20kilometers)+front(x,y,z,20kilometers))/3/5

set!(model;b=B)


simulation = Simulation(model, Δt = 1minutes, stop_time = 10hours)

simulation.output_writers[:fields] =
    NetCDFOutputWriter(model, merge(model.velocities, model.tracers), filepath = "output.nc",
                     schedule=TimeInterval(30minute))


run!(simulation)
