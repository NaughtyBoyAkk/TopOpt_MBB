### reference -> Sigmund, O. A 99 line topology optimization code written in Matlab.
### Structural and Multidisciplinary Optimization, 2014, 21(2): 120-127.
###
### Objective function
function Objfun(xval::Vector{Float64}, nelx::Int, nely::Int, penal::Float64, rmin::Float64)
    #
    x = reshape(xval,(nely,nelx))
    U = FEAnalysis(nelx,nely,x,penal)
    dcdx = zeros(Float64, nely, nelx)
    c = 0.0
    Ke = KE()
    for ely = 1:nely
        for elx = 1:nelx
            n1 = (nely+1)*(elx-1)+ely
            n2 = (nely+1)* elx +ely
            Ue = U[ [2*n1-1;2*n1; 2*n2-1;2*n2; 2*n2+1;2*n2+2; 2*n1+1;2*n1+2] ]
            c = c + x[ely,elx]^penal * transpose(Ue)*Ke*Ue
            dcdx[ely,elx] = -penal*x[ely,elx]^(penal-1) * transpose(Ue)*Ke*Ue
        end
    end
    dcdx_check = check(nelx,nely,rmin,x,dcdx)
    #
    grad = reshape(dcdx_check,nelx*nely)
    return c, grad
end
#
### FEM Analysis
function FEAnalysis(nelx::Int, nely::Int, x::Array{Float64,2}, penal::Float64)
    Ke = KE()
    K = spzeros(Float64, 2*(nelx+1)*(nely+1), 2*(nelx+1)*  (nely+1))
    Fext = zeros(Float64, 2*(nely+1)*(nelx+1) )
    U    = zeros(Float64, 2*(nely+1)*(nelx+1) )
    for ely = 1 : nely
        for elx = 1 : nelx
            n1 = (nely+1) * (elx-1) + ely
            n2 = (nely+1) * elx + ely
            edof = [2*n1-1; 2*n1; 2*n2-1; 2*n2; 2*n2+1; 2*n2+2; 2*n1+1; 2*n1+2]
            K[edof,edof] = K[edof,edof] + (x[ely,elx]^penal).*Ke
        end
    end
    # DEFINE LOADSAND SUPPORTS(HALF MBB-BEAM)
    Fext[2] = -1.0
    fixeddofs = union(range(1,2*(nely+1),step=2), 2*(nelx+1)*(nely+1))
    alldofs = Vector( 1:2*(nely+1)*(nelx+1) )
    freedofs = setdiff( alldofs, fixeddofs )
    # SOLVING
    U[freedofs] = K[freedofs,freedofs] \ Fext[freedofs]
    U[fixeddofs] .= 0.0
    return U
end
#
### Element stiffness
function KE()
    EMAT = 1.0
    nu = 0.3
    k=[ 1.0/2.0 - nu/6.0; 1.0/8.0 + nu/8.0; -1.0/4.0 - nu/12.0; -1.0/8.0 + 3.0*nu/8.0;
        -1.0/4.0 + nu/12.0; -1.0/8.0 - nu/8.0; nu/6.0; 1.0/8.0 - 3.0*nu/8.0]
    Ke = EMAT/[1.0 - nu^2] .*
       [ k[1] k[2] k[3] k[4] k[5] k[6] k[7] k[8];
         k[2] k[1] k[8] k[7] k[6] k[5] k[4] k[3];
         k[3] k[8] k[1] k[6] k[7] k[4] k[5] k[2];
         k[4] k[7] k[6] k[1] k[8] k[3] k[2] k[5];
         k[5] k[6] k[7] k[8] k[1] k[2] k[3] k[4];
         k[6] k[5] k[4] k[3] k[2] k[1] k[8] k[7];
         k[7] k[4] k[5] k[2] k[3] k[8] k[1] k[6];
         k[8] k[3] k[2] k[5] k[4] k[7] k[6] k[1] ]
    return Ke
end
#
### Filter
function check(nelx::Int,nely::Int,rmin::Float64,x::Array{Float64,2},dc::Array{Float64,2})
    dcn=zeros(Float64,nely,nelx)
    for i = 1 : nelx
        for j = 1 : nely
            sum = 0.0
            for k = max(i-round(Int,rmin),1) : min(i+round(Int,rmin),nelx)
                for l = max(j-round(Int,rmin),1) : min(j+round(Int,rmin), nely)
                    fac = rmin - sqrt( (i-k)^2 + (j-l)^2 )
                    sum = sum + max(0.0, fac)
                    dcn[j,i] = dcn[j,i] + max(0.0, fac) * x[l,k] * dc[l,k]
                end
            end
            dcn[j,i] = dcn[j,i] / (x[j,i] * sum)
        end
    end
    return dcn
end
#
### Constraints function
function Confun(x::Vector{Float64},volfrac::Float64)
    fc = [ sum(x)-volfrac*length(x) ]
    grad = zeros(Float64, 1, length(x))
    grad[:] .= 1.0
    return fc, grad
end
