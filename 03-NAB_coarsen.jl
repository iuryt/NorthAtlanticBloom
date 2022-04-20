using CUDA
using Oceananigans
using Oceananigans.Units

using NCDatasets
ds = Dataset("data/input_coarsen.nc")
Nx, Ny, Nz = size(ds["b"])


grid = RectilinearGrid(GPU(),
    size=(Nx, Ny, Nz),
    x=(minimum(ds["xC"]), maximum(ds["xC"])),
    y=(minimum(ds["yC"]), maximum(ds["yC"])),
    z=ds["zF"][:],
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


@inline νh(x,y,z,t) = ifelse((y>-((Ny*10-20)/2)kilometers)&(y<((Ny*10-20)/2)kilometers), 10, 200)
horizontal_closure = HorizontalScalarDiffusivity(ν=νh, κ=νh)

@inline νz(x,y,z,t) = ifelse((y>-((Ny*10-20)/2)kilometers)&(y<((Ny*10-20)/2)kilometers), 1e-5, 1e-4)
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


using KernelAbstractions: @index, @kernel
using KernelAbstractions.Extras.LoopInfo: @unroll
using Oceananigans.Architectures: device_event, architecture
using Oceananigans.Utils: launch!
using Oceananigans.Grids


@kernel function _replace_values!(target_field, source_field)
    i, j, k = @index(Global, NTuple)

    @inbounds target_field[i, j, k] = source_field[i, j, k]
    
end

function replace_values!(target_field, source_field)
    grid = target_field.grid
    arch = architecture(grid)

    event = launch!(arch, grid, :xyz,
                    _replace_values!, target_field, source_field,
                    dependencies = device_event(arch))

    wait(device_event(arch), event)

    return nothing
end

b = model.tracers.b
replace_values!(b, CuArray{Float64}(ds["b"][:]))

ui = CenterField(grid)
replace_values!(ui, CuArray{Float64}(ds["u"][:]))
ui = @at((Face, Center, Center), ui)


vi = CenterField(grid)
replace_values!(vi, CuArray{Float64}(ds["v"][:]))
vi = @at((Center, Face, Center), vi)

u, v, w = model.velocities
replace_values!(u, ui)
replace_values!(v, vi)


simulation = Simulation(model, Δt = 1minutes, stop_time = (80-25)day)

wizard = TimeStepWizard(cfl=1.0, max_change=1.1, max_Δt=6minutes)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))


simulation.output_writers[:fields] =
    NetCDFOutputWriter(model, merge(model.velocities, model.tracers), filename = "data/output_coarsen.nc",
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
