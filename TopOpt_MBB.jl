########################################################################################################
### A Julia code for 2-D MBB beam topology optimization using MMA
########################################################################################################

########################################################################################################
### LOADING MODULES                                                                                  ###
########################################################################################################

# Loading modules
using Printf
using LinearAlgebra
using SparseArrays
using Plots
push!(LOAD_PATH, "./~")
include("MMA.jl")
include("funcs.jl")

########################################################################################################
### MAIN FUNCTION                                                                                    ###
########################################################################################################

function main(nelx::Int, nely::Int, volfrac::Float64, penal::Float64, rmin::Float64)
    # Beam initial settings
    n = nelx * nely
    m = 1
    eeen = ones(Float64, n)
    eeem = ones(Float64, m)
    zeron = zeros(Float64, n)
    zerom = zeros(Float64, m)
    xval = 0.5 .* copy(eeen)
    xold1 = copy(xval)
    xold2 = copy(xval)
    xmin = 0.001 .* eeen
    xmax = 1.0 .* eeen
    low = copy(xmin)
    upp = copy(xmax)
    move = 1.0
    c = 1.0e6 .* copy(eeem)
    d = copy(eeem)
    a0 = 1.0
    a = copy(zerom)
    # The iterations starts
    xchange = 1.0
    minxchange = 0.001
    outit = 0
    maxoutit = 120
    anim = @animate while (xchange > minxchange) && (outit < maxoutit)
        outit += 1
        # Calculate function values and gradients of the objective and constraints functions
        f0val, df0dx =  Objfun(xval, nelx, nely, penal, rmin)
        fval, dfdx = Confun(xval, volfrac)
        # The MMA subproblem is solved at the point xval:
        xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,low,upp =
            mmasub(m,n,outit,xval,xmin,xmax,xold1,xold2,f0val,df0dx,fval,dfdx,low,upp,a0,a,c,d,move)
        # Some vectors are updated:
        xold2 = copy(xold1)
        xold1 = copy(xval)
        xval = copy(xmma)
        # Compute the change by the inf. norm
        xchange = norm( xval - xold1, Inf)
        # Write iteration history to screen
        @printf("\n It.: %3d, obj.: %.3f vol.: %.3f, ch.: %.3f \n", outit, f0val, sum(xval)/n, xchange )
        # plot map
        xplot = reverse(reshape(xval,(nely,nelx)),dims=1)
        p = heatmap(
        xplot,
        color = :Greys_9,
        clim = (0.0, 1.0),
        aspect_ratio = 1,
        xaxis = (0:10:nelx),
        yaxis = (0:10:nely)
        )
        display(p)
    end
    gif(anim, "Topologyhistory.gif",fps = 8)
    return xval
end

########################################################################################################
### RUN MAIN FUNCTION                                                                                ###
########################################################################################################

# Run main function / program
nelx = 150
nely = 50
penal = 3.0
rmin = 1.5
volfrac = 0.5
#
xval = main(nelx, nely, volfrac, penal, rmin)
