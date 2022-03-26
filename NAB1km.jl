using Oceananigans
using Oceananigans.Units

const sponge=10 #number of points for sponge
const Nx = 96 # number of points in x
const Ny = 96 # number of points in y
const Nz = 48 # number of points in z
const H = 1000 # maximum depth


grid = RectilinearGrid(GPU(),
    size=(Nx+2sponge,Ny,Nz),
    halo=(3,3,3),
    x=(-(Nx/2 + sponge)kilometers, (Nx/2 + sponge)kilometers), 
    y=(-(Ny/2)kilometers, (Ny/2)kilometers), 
    z=(H * cos.(LinRange(π/2,0,Nz+1)) .- H)meters,
    topology=(Bounded, Periodic, Bounded)
)

coriolis = FPlane(latitude=60)

@inline νh(x,y,z,t) = ifelse((x>-(Nx/2)kilometers)&(x<(Nx/2)kilometers), 1, 10)
horizontal_closure = HorizontalScalarDiffusivity(ν=νh, κ=νh)

@inline νz(x,y,z,t) = ifelse((x>-(Nx/2)kilometers)&(x<(Nx/2)kilometers), 1e-5, 1e-4)
vertical_closure = ScalarDiffusivity(ν=νz, κ=νz)


growing_and_grazing(x, y, z, t, P, params) = (params.μ₀ * exp(z / params.λ) - params.m) * P
plankton_dynamics_parameters = (μ₀ = 1/day,   # surface growth rate
                                 λ = 5,       # sunlight attenuation length scale (m)
                                 m = 0.1/day) # mortality rate due to virus and zooplankton grazing

plankton_dynamics = Forcing(growing_and_grazing, field_dependencies = :P,
                            parameters = plankton_dynamics_parameters)


    
model = NonhydrostaticModel(grid = grid,
                            advection = UpwindBiasedFifthOrder(),
                            coriolis = coriolis,
                            closure=(horizontal_closure,vertical_closure),
                            tracers = (:b, :P), # P for Plankton
                            buoyancy = BuoyancyTracer(),
                            forcing = (P=plankton_dynamics,))



const cz = -250meters # mld 
const L = 10kilometers
const amp = 1kilometers
@inline front(x, y, z, cx) = ((tanh(0.03*(z-cz))+1)/2)*(tanh(0.5*(x-(cx+sin(2pi*y/L)*amp)))+1)/2
@inline B(x, y, z) = (front(x,y,z,-20kilometers)+front(x,y,z,0kilometers)+front(x,y,z,20kilometers))/3/5e2
@inline P(x, y, z) = ifelse(z>cz,0.4,0)

set!(model;b=B,P=P)


simulation = Simulation(model, Δt = 1minutes, stop_time = 30day)


wizard = TimeStepWizard(cfl=1.0, max_change=1.1, max_Δt=2minutes)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))


simulation.output_writers[:fields] =
    NetCDFOutputWriter(model, merge(model.velocities, model.tracers), filepath = "output.nc",
                     schedule=TimeInterval(1day))


using Printf

function print_progress(simulation)
    u, v, w = simulation.model.velocities

    # Print a progress message
    msg = @sprintf("i: %04d, t: %s, Δt: %s, umax = (%.1e, %.1e, %.1e) ms⁻¹, wall time: %s\n",
                   iteration(simulation),
                   prettytime(time(simulation)),
                   prettytime(simulation.Δt),
                   maximum(abs, u), maximum(abs, v), maximum(abs, w),
                   prettytime(simulation.run_wall_time))

    @info msg

    return nothing
end

simulation.callbacks[:progress] = Callback(print_progress, TimeInterval(1hour))

run!(simulation)
