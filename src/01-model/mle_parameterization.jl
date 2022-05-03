# repeating the imports just to make it easier while moduling it
using KernelAbstractions: @index, @kernel
using KernelAbstractions.Extras.LoopInfo: @unroll
using Oceananigans.Architectures: device_event, architecture
using Oceananigans.Utils: launch!
using Oceananigans.Grids
using Oceananigans.Operators: Δzᶜᶜᶜ, Δzᶜᶜᶠ


@kernel function _compute_Ψₑ!(Ψₑ, grid, h, ∂ₕb, ∂b∂z, μ, f, ce, Lfₘ, ΔS, τ, minpoints, Vm)
    i, j = @index(Global, NTuple)

    
    # average ∇ₕb and N over the mixed layer
    
    ∂ₕb_sum = 0
    N_sum = 0
    Δz_sum = 0

    h_ij = @inbounds h[i, j]
    
    # number of z points in the mixed layer
    npoints = 0
    
    @unroll for k in grid.Nz : -1 : 1 # scroll from surface to bottom       
        z_center = znode(Center(), Face(), Face(), i, j, k, grid)

        if z_center > -h_ij
            npoints += 1
            
            Δz_ijk = Δzᶜᶜᶜ(i, j, k, grid)

            ∂ₕb_sum = ∂ₕb_sum + @inbounds ∂ₕb[i, j, k] * Δz_ijk
            N_sum = N_sum + @inbounds sqrt(max(zero(eltype(grid)), ∂b∂z[i, j, k])) * Δz_ijk 
            Δz_sum = Δz_sum + Δz_ijk

        end
    end

    ∂ₕbₘₗ = ∂ₕb_sum/Δz_sum
    Nₘₗ = N_sum/Δz_sum
    
    Lf = max(Nₘₗ*h_ij/abs(f), Lfₘ)
    
    # compute eddy stream function
    @unroll for k in grid.Nz : -1 : 1 # scroll to point just above the bottom       
        z_face = znode(Center(), Center(), Face(), i, j, k, grid)
        
        Ψ_max = @inbounds Δzᶜᶜᶠ(i, j, k, grid) * Vm
        Ψ_ijk = ce * (ΔS/Lf) * ((h_ij^2)/sqrt(f^2 + τ^-2)) * μ(z_face,h_ij) * ∂ₕbₘₗ
        
        if (z_face > -h_ij) & (npoints > minpoints)
            @inbounds Ψₑ[i, j, k] = min(Ψ_max, Ψ_ijk)
        else
            @inbounds Ψₑ[i, j, k] = 0.0
        end
    end

end

function compute_Ψₑ!(Ψₑ, h, ∂ₕb, ∂b∂z, μ, f; ce = 0.06, Lfₘ = 500meters, ΔS=10kilometers, τ=86400, minpoints=4, Vm=0.5)
    grid = h.grid
    arch = architecture(grid)


    event = launch!(arch, grid, :xy,
                    _compute_Ψₑ!, Ψₑ, grid, h, ∂ₕb, ∂b∂z, μ, f, ce, Lfₘ, ΔS, τ, minpoints, Vm,
                    dependencies = device_event(arch))

    wait(device_event(arch), event)
    
    fill_halo_regions!(Ψₑ, arch)
    return nothing
end