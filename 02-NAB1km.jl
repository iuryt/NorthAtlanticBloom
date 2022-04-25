using Bioceananigans
using Oceananigans
using Oceananigans.Units

const sponge = 20 #number of points for sponge
const Nx = 100 # number of points in x
const Ny = 460 # number of points in y
const Nz = 48 # number of points in z
const H = 1000 # maximum depth


grid = RectilinearGrid(GPU(),
    size=(Nx,Ny+2sponge,Nz),
    halo=(3,3,3),
    x=(-(Nx/2)kilometers, (Nx/2)kilometers), 
    y=(-(Ny/2 + sponge)kilometers, (Ny/2 + sponge)kilometers), 
    z=(H * cos.(LinRange(π/2,0,Nz+1)) .- H)meters,
    topology=(Periodic, Bounded, Bounded)
)

coriolis = FPlane(latitude=60)

cᴰ = 1e-4 # quadratic drag coefficient

@inline bottom_drag_u(x, y, t, u, v, cᴰ) = - cᴰ * u * sqrt(u^2 + v^2)
@inline bottom_drag_v(x, y, t, u, v, cᴰ) = - cᴰ * v * sqrt(u^2 + v^2)

bottom_drag_bc_u = FluxBoundaryCondition(bottom_drag_u, field_dependencies=(:u, :v), parameters=cᴰ)
bottom_drag_bc_v = FluxBoundaryCondition(bottom_drag_v, field_dependencies=(:u, :v), parameters=cᴰ)

u_bcs = FieldBoundaryConditions(bottom = bottom_drag_bc_u)
v_bcs = FieldBoundaryConditions(bottom = bottom_drag_bc_v)


@inline νh(x,y,z,t) = ifelse((y>-(Ny/2)kilometers)&(y<(Ny/2)kilometers), 1, 120)
horizontal_closure = HorizontalScalarDiffusivity(ν=νh, κ=νh)

@inline νz(x,y,z,t) = ifelse((y>-(Ny/2)kilometers)&(y<(Ny/2)kilometers), 1e-5, 1e-4)
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
                            tracers = (:b),
                            buoyancy = BuoyancyTracer(),
                            boundary_conditions = (u=u_bcs, v=v_bcs))



const L = (Nx-1)kilometers/10
const amp = 1kilometers
const g = 9.82
const ρₒ = 1026

# background density profile based on Argo data
@inline bg(z) = 0.08*tanh(0.005*(-618-z))+0.014*(z^2)/1e5+1027.45

# decay function for fronts
@inline decay(z) = (tanh((z+500)/300)+1)/2

# front function
@inline front(x, y, z, cy) = tanh((y-(cy+sin(2π * x / L)*amp))/12kilometers)

@inline D(x, y, z) = bg(z) + 0.8*decay(z)*((front(x, y, z, -120kilometers)+front(x, y, z, 0)+front(x, y, z, 120kilometers))-3)/6
@inline B(x, y, z) = -(g/ρₒ)*D(x, y, z)

set!(model; b = B)


b = model.tracers.b
f = model.coriolis.f

# shear operations
uz_op = @at((Face, Center, Center), -∂y(b) / f );
vz_op = @at((Center, Face, Center),  ∂x(b) / f );
# compute shear
uz = compute!(Field(uz_op))
vz = compute!(Field(vz_op))

# include function for cumulative integration
include("src/cumulative_vertical_integration.jl")

# compute geostrophic velocities
U = cumulative_vertical_integration!(uz)
V = cumulative_vertical_integration!(vz)

# prescribe geostrophic velocities for initial condition
set!(model; u = U, v = V)


simulation = Simulation(model, Δt = 1minutes, stop_time = 90day)

wizard = TimeStepWizard(cfl=1.0, max_change=1.1, max_Δt=6minutes)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))



h = Field{Center, Center, Nothing}(grid) 
# buoyancy decrease criterium for determining the mixed-layer depth
const Δb = (g/ρₒ) * 0.03
compute_mixed_layer_depth!(simulation) = MixedLayerDepth!(h, simulation.model.tracers.b, Δb)
# add the function to the callbacks of the simulation
simulation.callbacks[:compute_mld] = Callback(compute_mixed_layer_depth!)


simulation.output_writers[:fields] =
    NetCDFOutputWriter(model, merge(model.velocities, model.tracers, (; h,)), filename = "data/output_1km.nc",
                     schedule=TimeInterval(8hours))


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
