using Bioceananigans
using Oceananigans
using Oceananigans.Units
using Oceananigans.BoundaryConditions: ImpenetrableBoundaryCondition, fill_halo_regions!

using NCDatasets
ds = Dataset("../../data/interim/input_coarse.nc")
const Nx, Ny, Nz = size(ds["b"])
const initial_time = ds["time"][1]

grid = RectilinearGrid(GPU(),
    size=(Nx, Ny, Nz),
    x=(minimum(ds["xC"]), maximum(ds["xC"])),
    y=(minimum(ds["yC"]), maximum(ds["yC"])),
    z=ds["zF"][:],
    topology=(Periodic, Bounded, Bounded)
)

coriolis = FPlane(latitude=60)

const cᴰ = 1e-4 # quadratic drag coefficient

@inline bottom_drag_u(x, y, t, u, v, cᴰ) = - cᴰ * u * sqrt(u^2 + v^2)
@inline bottom_drag_v(x, y, t, u, v, cᴰ) = - cᴰ * v * sqrt(u^2 + v^2)

bottom_drag_bc_u = FluxBoundaryCondition(bottom_drag_u, field_dependencies=(:u, :v), parameters=cᴰ)
bottom_drag_bc_v = FluxBoundaryCondition(bottom_drag_v, field_dependencies=(:u, :v), parameters=cᴰ)

u_bcs = FieldBoundaryConditions(bottom = bottom_drag_bc_u)
v_bcs = FieldBoundaryConditions(bottom = bottom_drag_bc_v)


horizontal_closure = HorizontalScalarDiffusivity(ν=10, κ=10)
vertical_closure = ScalarDiffusivity(ν=1e-5, κ=1e-5)


# ----
# ---- sponges

const sponge_size = 100kilometers
const slope = 10kilometers
const ymin = minimum(ynodes(Center, grid))
const ymax = maximum(ynodes(Center, grid))


mask_func(x,y,z) = ((
     tanh((y-(ymax-sponge_size))/slope)
    *tanh((y-(ymin+sponge_size))/slope)
)+1)/2

mom_sponge = Relaxation(rate=1/1hour, mask=mask_func, target=0)

# ---- 
# ---- initial mixed-layer eddy velocities


model = NonhydrostaticModel(grid = grid,
                            advection = WENO5(),
                            coriolis = coriolis,
                            closure=(horizontal_closure,vertical_closure),
                            tracers = (:b),
                            buoyancy = BuoyancyTracer(),
                            forcing = (; u=mom_sponge, v=mom_sponge, w=mom_sponge),
                            boundary_conditions = (u=u_bcs, v=v_bcs))

model.clock.time = initial_time*days

bᵢ = Float64.(ds["b"][:])

set!(model; b=bᵢ)


b = model.tracers.b
f = model.coriolis.f

# shear operations
uz_op = @at((Face, Center, Center), -∂y(b) / f );
vz_op = @at((Center, Face, Center),  ∂x(b) / f );
# compute shear
uz = compute!(Field(uz_op))
vz = compute!(Field(vz_op))

# include function for cumulative integration
include("cumulative_vertical_integration.jl")

# compute geostrophic velocities
uᵢ = cumulative_vertical_integration!(uz)
vᵢ = cumulative_vertical_integration!(vz)

# prescribe geostrophic velocities for initial condition
set!(model; u = uᵢ, v = vᵢ)


simulation = Simulation(model, Δt = 1minutes, stop_time = 80day)

wizard = TimeStepWizard(cfl=1.0, max_change=1.1, max_Δt=1hour)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))


h = Field{Center, Center, Nothing}(grid) 
# buoyancy decrease criterium for determining the mixed-layer depth
const g = 9.82 # gravity
const ρₒ = 1026 # reference density
const Δb = (g/ρₒ) * 0.03
compute_mixed_layer_depth!(simulation) = MixedLayerDepth!(h, simulation.model.tracers.b, Δb)
# add the function to the callbacks of the simulation
simulation.callbacks[:compute_mld] = Callback(compute_mixed_layer_depth!)




outputs = merge(model.velocities, model.tracers, (;  h)) # make a NamedTuple with all outputs

simulation.output_writers[:fields] =
    NetCDFOutputWriter(model, outputs, filename = "../../data/raw/output_coarse_no_mle.nc",
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