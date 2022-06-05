using Bioceananigans
using Oceananigans
using Oceananigans.Units
using Oceananigans.BoundaryConditions: ImpenetrableBoundaryCondition
using KernelAbstractions.Extras.LoopInfo: @unroll

fronts = false

#--------------- Grid

const Nx = 100 # number of points in x
const Ny = 500 # number of points in y
const Nz = 48 # number of points in z
const H = 1000 # maximum depth


grid = RectilinearGrid(GPU(),
    size=(Nx,Ny,Nz),
    halo=(3,3,3),
    x=(-(Nx/2)kilometers, (Nx/2)kilometers), 
    y=(-(Ny/2)kilometers, (Ny/2)kilometers), 
    z=(H * cos.(LinRange(π/2,0,Nz+1)) .- H)meters,
    topology=(Periodic, Bounded, Bounded)
)

#--------------- Coriolis

coriolis = FPlane(latitude=60)

#--------------- Bottom drag

cᴰ = 1e-4 # quadratic drag coefficient

@inline bottom_drag_u(x, y, t, u, v, cᴰ) = - cᴰ * u * sqrt(u^2 + v^2)
@inline bottom_drag_v(x, y, t, u, v, cᴰ) = - cᴰ * v * sqrt(u^2 + v^2)

bottom_drag_bc_u = FluxBoundaryCondition(bottom_drag_u, field_dependencies=(:u, :v), parameters=cᴰ)
bottom_drag_bc_v = FluxBoundaryCondition(bottom_drag_v, field_dependencies=(:u, :v), parameters=cᴰ)

u_bcs = FieldBoundaryConditions(bottom = bottom_drag_bc_u)
v_bcs = FieldBoundaryConditions(bottom = bottom_drag_bc_v)


#--------------- Sponges

const sponge_size = 50kilometers
const slope = 10kilometers
const ymin = minimum(ynodes(Center, grid))
const ymax = maximum(ynodes(Center, grid))

@inline mask_func(x,y,z) = ((
     tanh((y-(ymax-sponge_size))/slope)
    *tanh((y-(ymin+sponge_size))/slope)
)+1)/2

mom_sponge = Relaxation(rate=1/1hour, mask=mask_func, target=0)

horizontal_closure = HorizontalScalarDiffusivity(ν=1, κ=1)
vertical_closure = ScalarDiffusivity(ν=1e-5, κ=1e-5)

# @inline horizontal_closure_func(x,y,z,t) = 1 + mask_func(x,y,z) * (120 - 1)
# @inline vertical_closure_func(x,y,z,t) = 1e-5 + mask_func(x,y,z) * (1e-4 - 1e-5)

# horizontal_closure = HorizontalScalarDiffusivity(ν=horizontal_closure_func, κ=horizontal_closure_func)
# vertical_closure = ScalarDiffusivity(ν=vertical_closure_func, κ=vertical_closure_func)


#--------------- NP Model

# constants for the NP model
const μ₀ = 1/day   # surface growth rate
const m = 0.015/day # mortality rate due to virus and zooplankton grazing
const Kw = 0.059 # meter^-1
const kn = 0.75
const kr = 0.5

#  https://doi.org/10.1029/2017GB005850
const chl2c = 0.06 # average value for winter in North Atlantic

const α = 0.0538/day

const average = :growth
const shading = true

# create the mld field that will be updated at every timestep
h = Field{Center, Center, Nothing}(grid) 
light_growth = Field{Center, Center, Center}(grid)


# time evolution of shortwave radiation (North Atlantic)
@inline Lₒ(t) = 116 * sin( 2π * ( t / days + 50 ) / 375.3 - 1.3 ) + 132.3
# evolution of the available light at the surface
@inline light_function(t, z) = 0.43 * Lₒ(t) * exp( z * Kw )
# light profile
@inline light_growth_function(light) = μ₀ * ( light * α ) / sqrt( μ₀^2 + ( light * α )^2 )


# nitrate and ammonium limiting
@inline N_lim(N, Nr) = (N/(N+kn)) * (kr/(Nr+kr))
@inline Nr_lim(Nr) =  (Nr/(Nr+kr))

# functions for the NP model
@inline P_forcing(light_growth, P, N, Nr)  =   light_growth * (N_lim(N, Nr) + Nr_lim(Nr)) * P - m * P^2
@inline N_forcing(light_growth, P, N, Nr)  = - light_growth * N_lim(N, Nr) * P
@inline Nr_forcing(light_growth, P, N, Nr) = - light_growth * Nr_lim(Nr) * P + m * P^2

# functions for the NP model
@inline P_forcing(i, j, k, grid, clock, fields, p)  = @inbounds P_forcing(p.light_growth[i, j, k], fields.P[i, j, k], fields.N[i, j, k], fields.Nr[i, j, k])
@inline N_forcing(i, j, k, grid, clock, fields, p)  = @inbounds N_forcing(p.light_growth[i, j, k], fields.P[i, j, k], fields.N[i, j, k], fields.Nr[i, j, k])
@inline Nr_forcing(i, j, k, grid, clock, fields, p) = @inbounds Nr_forcing(p.light_growth[i, j, k], fields.P[i, j, k], fields.N[i, j, k], fields.Nr[i, j, k])

# using the functions to determine the forcing
P_dynamics = Forcing(P_forcing, discrete_form=true, parameters=(; light_growth))
N_dynamics = Forcing(N_forcing, discrete_form=true, parameters=(; light_growth))
Nr_dynamics = Forcing(Nr_forcing, discrete_form=true, parameters=(; light_growth))

# sinking velocity

# Vertical velocity function
const w_sink = -1meter/day
const lamb = 1meters
@inline w_func(x, y, z) = w_sink * tanh(max(-z / lamb, 0.0)) * tanh(max((z + H) / lamb, 0.0))

no_penetration = ImpenetrableBoundaryCondition()
w_bc = FieldBoundaryConditions(grid, (Center, Center, Face), top=no_penetration, bottom=no_penetration)

# Field (allocates memory and precalculates w_func)
w = Field((Center, Center, Face), grid, boundary_conditions=w_bc)
set!(w, w_func)
P_sink = AdvectiveForcing(WENO5(; grid), w = w)


#--------------- Instantiate Model

forcing = (;
    P=(P_dynamics, P_sink), N=N_dynamics, Nr=Nr_dynamics,
    u=mom_sponge, v=mom_sponge, w=mom_sponge,
)

model = NonhydrostaticModel(grid = grid,
                            advection = WENO5(),
                            coriolis = coriolis,
                            closure = (horizontal_closure,vertical_closure),
                            tracers = (:b, :P, :N, :Nr),
                            buoyancy = BuoyancyTracer(),
                            forcing = forcing,
                            boundary_conditions = (u=u_bcs, v=v_bcs))


#--------------- Initial Conditions

const L = (Nx-1)kilometers/10
const amp = 1kilometers
const g = 9.82
const ρₒ = 1026

# background density profile based on Argo data
@inline bg(z) = -0.147 * tanh( 2.6 * ( z + 623 ) ) - 1027.6

# decay function for fronts
@inline decay(z) = ( tanh( (z + 500) / 300) + 1 ) / 2

# front function
@inline front(x, y, z, cy) = tanh( ( y - ( cy + sin(2π * x / L) * amp ) ) / 12kilometers )


@inline D(x, y, z) = bg(z) + fronts*0.8*decay(z)*((front(x, y, z, -100kilometers)+front(x, y, z, 0)+front(x, y, z, 100kilometers))-3)/6
@inline B(x, y, z) = -(g/ρₒ)*D(x, y, z)

# initial phytoplankton profile
@inline P(x, y, z) = 0.1 * ( tanh( 0.01 * (z + 300)) + 1) / 2

# initial nitrate profile
@inline N(x, y, z) = z * (12 - 16) / (0 + 800) + 12

# setting the initial conditions
set!(model; b=B, P=P, N=N, Nr=0)



#--------------- Initial Geostrophic Velocities

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
U = cumulative_vertical_integration!(uz)
V = cumulative_vertical_integration!(vz)

# prescribe geostrophic velocities for initial condition
set!(model; u = U, v = V)


#--------------- Simulation

simulation = Simulation(model, Δt = 1minutes, stop_time = 90day)

wizard = TimeStepWizard(cfl=1.0, max_change=1.1, max_Δt=1hour)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

#--------------- Mixed Layer Depth

h = Field{Center, Center, Nothing}(grid) 
# buoyancy decrease criterium for determining the mixed-layer depth
const Δb = (g/ρₒ) * 0.03
compute_mixed_layer_depth!(simulation) = MixedLayerDepth!(h, simulation.model.tracers.b, Δb)
# add the function to the callbacks of the simulation
simulation.callbacks[:compute_mld] = Callback(compute_mixed_layer_depth!)

#--------------- Light-limiting growth

compute_light_growth!(simulation) = LightGrowth!(light_growth, h, simulation.model.tracers.P, light_function, light_growth_function, time(simulation), average, shading, chl2c)
# add the function to the callbacks of the simulation
simulation.callbacks[:compute_light_growth] = Callback(compute_light_growth!)

#--------------- Zeroing negative values

# zeroing negative values
@inline function zeroing(sim)
    @unroll for tracer in [:P, :N, :Nr]
        parent(sim.model.tracers[tracer]) .= max.(0, parent(sim.model.tracers[tracer]))
    end
end
simulation.callbacks[:zeroing] = Callback(zeroing)

#--------------- Writing Outputs

bi, Pi, Ni, Nri = simulation.model.tracers
u, v, w = simulation.model.velocities


extra_outputs = (; 
    h=h, 
    light_growth=light_growth, 
    new_production=light_growth * N_lim(Ni, Nri) * Pi,
    u=@at((Center, Center, Center), u),
    v=@at((Center, Center, Center), v),
    w=@at((Center, Center, Center), w),
    N2=@at((Center, Center, Center), ∂z(bi)),
    ∇b=@at((Center, Center, Center), sqrt(∂x(bi)^2 + ∂y(bi)^2)),
    Ro=@at((Center, Center, Center), (∂x(v)-∂y(u))/f),
)

if fronts
    simulation.output_writers[:fields] =
        NetCDFOutputWriter(model, merge(model.tracers, extra_outputs),
                         overwrite_existing=true,
                         filename = "../../data/raw/output_submesoscale.nc",
                         schedule=TimeInterval(8hours))
else
    simulation.output_writers[:fields] =
        NetCDFOutputWriter(model, merge(model.tracers, extra_outputs),
                         overwrite_existing=true,
                         filename = "../../data/raw/output_no_fronts.nc",
                         schedule=TimeInterval(8hours))
end

#--------------- Printing Progress

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

simulation.callbacks[:progress] = Callback(print_progress, TimeInterval(12hour))

run!(simulation)
