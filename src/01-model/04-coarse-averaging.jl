using Bioceananigans
using Oceananigans
using Oceananigans.Units
using Oceananigans.BoundaryConditions: ImpenetrableBoundaryCondition, fill_halo_regions!
using KernelAbstractions.Extras.LoopInfo: @unroll


sinking = true

#--------------- Grid

using NCDatasets
ds = Dataset("../../data/interim/input_coarse.nc")

const H = 1000 # maximum depth



const Nx, Ny, Nz = size(ds["b"][:,:,:,1])

grid = RectilinearGrid(GPU(),
    size=(Nx, Ny, Nz),
    x=(minimum(ds["xF"]),maximum(ds["xF"])+diff(ds["xF"])[end]),
    y=(minimum(ds["yF"]),maximum(ds["yF"])+diff(ds["yF"])[end]),
    z=ds["zF"][:],
    topology=(Periodic, Bounded, Bounded)
)


#--------------- Coriolis

coriolis = FPlane(latitude=60)


#--------------- Sponges

horizontal_closure = HorizontalScalarDiffusivity(ν=10, κ=10)
vertical_closure = ScalarDiffusivity(ν=1e-5, κ=1e-5)

#--------------- NP Model

# constants for the NP model
const μ₀ = 2/day   # surface growth rate
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
const w_sink = ifelse(sinking,-1meter/day,0)
const lamb = 1meters
@inline w_func(x, y, z) = w_sink * tanh(max(-z / lamb, 0.0)) * tanh(max((z + H) / lamb, 0.0))

no_penetration = ImpenetrableBoundaryCondition()
w_bc = FieldBoundaryConditions(grid, (Center, Center, Face), top=no_penetration, bottom=no_penetration)

# Field (allocates memory and precalculates w_func)
w = Field((Center, Center, Face), grid, boundary_conditions=w_bc)
set!(w, w_func)
P_sink = AdvectiveForcing(WENO5(; grid), w = w)



#--------------- Instantiate Model

no_penetration = ImpenetrableBoundaryCondition()
v_bc = FieldBoundaryConditions(grid, (Center, Face, Center), north=no_penetration, south=no_penetration)
w_bc = FieldBoundaryConditions(grid, (Center, Center, Face), top=no_penetration, bottom=no_penetration)

u = Field((Face, Center, Center), grid)
v = Field((Center, Face, Center), grid, boundary_conditions=v_bc)
w = Field((Center, Center, Face), grid, boundary_conditions=w_bc)

forcing = (;
    P=(P_dynamics, P_sink), N=(N_dynamics), 
    Nr=(Nr_dynamics),
)


model = HydrostaticFreeSurfaceModel(grid = grid,
                            tracer_advection = WENO5(),
                            momentum_advection = nothing,
                            coriolis = coriolis,
                            velocities = PrescribedVelocityFields(u=u, v=v, w=w),
                            buoyancy = nothing,
                            forcing = forcing,
                            closure=(horizontal_closure,vertical_closure),
                            tracers = (:b, :P, :N, :Nr))


bᵢ = Float64.(ds["b"][:,:,:,1])
Pᵢ = Float64.(ds["P"][:,:,:,1])
Nᵢ = Float64.(ds["N"][:,:,:,1])
Nrᵢ = Float64.(ds["Nr"][:,:,:,1])

set!(model; b=bᵢ, P=Pᵢ, N=Nᵢ, Nr=Nrᵢ)

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
uᵢ = cumulative_vertical_integration!(uz)
vᵢ = cumulative_vertical_integration!(vz)

set!(model; u=uᵢ, v=vᵢ)

#--------------- Simulation

simulation = Simulation(model, Δt = 3hour, stop_time = 90day)


#--------------- Mixed Layer Depth
const g = 9.82
const ρₒ = 1026

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

#--------------- Prescribe buoyancy


function update_b!(sim)
    ti = sim.model.clock.time
    
    set!(model,
        b = Float64.(ds["b"][:,:,:,Int(round(ti/3hours))+1])
    )
    return nothing
end
simulation.callbacks[:update_b] = Callback(update_b!)


#--------------- Compute geostrophic velocities

# include function for cumulative integration
include("cumulative_vertical_integration.jl")

function compute_uv!(sim)
    b = model.tracers.b
    f = model.coriolis.f

    # shear operations
    uz_op = Field(@at((Face, Center, Center), -∂y(b) / f ));
    vz_op = Field(@at((Center, Face, Center),  ∂x(b) / f ));
    
    uᵢ = compute!(cumulative_vertical_integration!(uz))
    vᵢ = compute!(cumulative_vertical_integration!(vz))
    
    set!(model; u=uᵢ, v=vᵢ)
    return nothing
end
simulation.callbacks[:compute_uv] = Callback(compute_uv!)



#--------------- Writing Outputs

bi, Pi, Ni, Nri = simulation.model.tracers
u = simulation.model.velocities.u
v = simulation.model.velocities.v
w = simulation.model.velocities.w


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
    hx=@at((Center, Face, Nothing), h), 
    hy=@at((Face, Center, Nothing), h), 
)

if sinking
    filename = "../../data/raw/output_coarse_averaging_mu$(μ₀*days).nc"
else
    filename = "../../data/raw/output_coarse_averaging_nosinking_mu$(μ₀*days).nc"
end

simulation.output_writers[:fields] =
    NetCDFOutputWriter(model, merge(model.tracers, extra_outputs), 
                    filename = filename,
                    overwrite_existing=true,
                    schedule=IterationInterval(1))

#--------------- Printing Progress

using Printf

function print_progress(simulation)
    u = simulation.model.velocities.u
    v = simulation.model.velocities.v
    w = simulation.model.velocities.w

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