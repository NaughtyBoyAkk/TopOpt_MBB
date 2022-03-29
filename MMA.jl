########################################################################################################
### GCMMA-MMA-Julia (1.7.0)         															     ###
###                                                                                                  ###
### This file is part of GCMMA-MMA-Julia (1.7.0).                                                    ###
###                                                                                                  ###
### The orginal work is written by Krister Svanberg in MATLAB.                                       ###
### This is the Julia (1.7.0) version of the code written by Ruxinakk.                               ###
### version 03-29-2022                                                                               ###                                                                                                  ###
### Modified from  Python version written by Arjen Deetman                                           ###
### <https://github.com/arjendeetman/GCMMA-MMA-Python>                                               ###
########################################################################################################
#
# This file gives the functions mmasub, gcmmasub, subsolv and kktcheck.
#
########################################################################################################
### LOADING MODULES                                                                                  ###
########################################################################################################

# Loading modules
using LinearAlgebra
using SparseArrays

########################################################################################################
### MMA FUNCTIONS                                                                                    ###
########################################################################################################

# Function for the MMA sub problem
function mmasub(m::Int, n::Int, iter::Int, xval::Array{Float64,1}, xmin::Array{Float64,1},
    xmax::Array{Float64,1}, xold1::Array{Float64,1}, xold2::Array{Float64,1}, f0val::Float64,
    df0dx::Array{Float64,1}, fval::Array{Float64,1}, dfdx::Array{Float64,2}, low::Array{Float64,1},
    upp::Array{Float64,1}, a0::Float64, a::Array{Float64,1}, c::Array{Float64,1}, d::Array{Float64,1},
    move::Float64)

    # """
    # This function mmasub performs one MMA-iteration, aimed at solving the nonlinear programming problem:
    #
    # Minimize    f_0(x) + a_0*z + sum( c_i*y_i + 0.5*d_i*(y_i)^2 )
    # subject to  f_i(x) - a_i*z - y_i <= 0,  i = 1,...,m
    #             xmin_j <= x_j <= xmax_j,    j = 1,...,n
    #             z >= 0,   y_i >= 0,         i = 1,...,m
    # INPUT:
    #
    #     m     = The number of general constraints.
    #     n     = The number of variables x_j.
    #     iter  = Current iteration number ( =1 the first time mmasub is called).
    #     xval  = Column vector with the current values of the variables x_j.
    #     xmin  = Column vector with the lower bounds for the variables x_j.
    #     xmax  = Column vector with the upper bounds for the variables x_j.
    #     xold1 = xval, one iteration ago (provided that iter>1).
    #     xold2 = xval, two iterations ago (provided that iter>2).
    #     f0val = The value of the objective function f_0 at xval.
    #     df0dx = Column vector with the derivatives of the objective function
    #             f_0 with respect to the variables x_j, calculated at xval.
    #     fval  = Column vector with the values of the constraint functions f_i, calculated at xval.
    #     dfdx  = (m x n)-matrix with the derivatives of the constraint functions
    #             f_i with respect to the variables x_j, calculated at xval.
    #             dfdx(i,j) = the derivative of f_i with respect to x_j.
    #     low   = Column vector with the lower asymptotes from the previous
    #             iteration (provided that iter>1).
    #     upp   = Column vector with the upper asymptotes from the previous
    #             iteration (provided that iter>1).
    #     a0    = The constants a_0 in the term a_0*z.
    #     a     = Column vector with the constants a_i in the terms a_i*z.
    #     c     = Column vector with the constants c_i in the terms c_i*y_i.
    #     d     = Column vector with the constants d_i in the terms 0.5*d_i*(y_i)^2.
    #
    # OUTPUT:
    #
    #     xmma  = Column vector with the optimal values of the variables x_j
    #             in the current MMA subproblem.
    #     ymma  = Column vector with the optimal values of the variables y_i
    #             in the current MMA subproblem.
    #     zmma  = Scalar with the optimal value of the variable z
    #             in the current MMA subproblem.
    #     lam   = Lagrange multipliers for the m general MMA constraints.
    #     xsi   = Lagrange multipliers for the n constraints alfa_j - x_j <= 0.
    #     eta   = Lagrange multipliers for the n constraints x_j - beta_j <= 0.
    #     mu    = Lagrange multipliers for the m constraints -y_i <= 0.
    #     zet   = Lagrange multiplier for the single constraint -z <= 0.
    #     s     = Slack variables for the m general MMA constraints.
    #     low   = Column vector with the lower asymptotes, calculated and used
    #             in the current MMA subproblem.
    #     upp   = Column vector with the upper asymptotes, calculated and used
    #             in the current MMA subproblem.
    # """

    epsimin = 1.0e-7
    raa0 = 1.0e-5
    albefa = 0.1
    asyinit = 0.5
    asyincr = 1.2
    asydecr = 0.7
    eeen = ones(Float64, n)
    eeem = ones(Float64, m)
    zeron = zeros(Float64, n)
    # Calculation of the asymptotes low and upp
    if iter <= 2
        low = xval - asyinit .* (xmax - xmin)
        upp = xval + asyinit .* (xmax - xmin)
    else
        zzz = (xval - xold1) .* (xold1 - xold2)
        factor = copy(eeen)
        factor[findall(zzz .> 0.0)] .= asyincr
        factor[findall(zzz .< 0.0)] .= asydecr
        low = xval - factor .* (xold1 - low)
        upp = xval + factor .* (upp - xold1)
        lowmin = xval - 10.0 .* (xmax - xmin)
        lowmax = xval - 0.01 .* (xmax - xmin)
        uppmin = xval + 0.01 .* (xmax - xmin)
        uppmax = xval + 10.0 .* (xmax - xmin)
        low = max.(low, lowmin)
        low = min.(low, lowmax)
        upp = min.(upp, uppmax)
        upp = max.(upp, uppmin)
    end
    # Calculation of the bounds alfa and beta
    zzz1 = low + albefa .* (xval - low)
    zzz2 = xval - move .* (xmax - xmin)
    zzz = max.(zzz1, zzz2)
    alfa = max.(zzz, xmin)
    zzz1 = upp - albefa .* (upp - xval)
    zzz2 = xval + move .* (xmax - xmin)
    zzz = min.(zzz1, zzz2)
    beta = min.(zzz, xmax)
    # Calculations of p0, q0, P, Q and b
    xmami = xmax - xmin
    xmamieps = 1.0e-5 .* eeen
    xmami = max.(xmami, xmamieps)
    xmamiinv = eeen ./ xmami
    ux1 = upp - xval
    ux2 = ux1 .* ux1
    xl1 = xval - low
    xl2 = xl1 .* xl1
    uxinv = eeen ./ ux1
    xlinv = eeen ./ xl1
    p0 = copy(zeron)
    q0 = copy(zeron)
    p0 = max.( df0dx, 0.0)
    q0 = max.(-df0dx, 0.0)
    pq0 = 0.001 .* (p0 + q0) + raa0 .* xmamiinv
    p0 = p0 + pq0
    q0 = q0 + pq0
    p0 = p0 .* ux2
    q0 = q0 .* xl2
    P = max.( dfdx, 0.0)
    Q = max.(-dfdx, 0.0)
    PQ = 0.001 .* (P+Q) + raa0 .* (eeem * transpose(xmamiinv))
    P = P + PQ
    Q = Q + PQ
    P = P .* transpose(ux2)
    Q = Q .* transpose(xl2)
    b = P * uxinv + Q * xlinv - fval
    # Solving the subproblem by a primal-dual Newton method
    xmma,ymma,zmma,lam,xsi,eta,mu,zet,s = subsolv(m,n,epsimin,low,upp,alfa,beta,p0,q0,P,Q,a0,a,b,c,d)
    # Return values
    return xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp
end

# Function for the GCMMA sub problem
function gcmmasub(m::Int, n::Int, epsimin::Float64, xval::Array{Float64,1}, xmin::Array{Float64,1},
    xmax::Array{Float64,1}, low::Array{Float64,1}, upp::Array{Float64,1}, raa0::Float64,
    raa::Array{Float64,1}, f0val::Float64, df0dx::Array{Float64,1}, fval::Array{Float64,1},
    dfdx::Array{Float64,2}, a0::Float64, a::Array{Float64,1}, c::Array{Float64,1}, d::Array{Float64,1})
    #
    eeen = ones(Float64, n)
    zeron = ones(Float64, n)
    # Calculations of the bounds alfa and beta
    albefa = 0.1
    zzz = low + albefa .* (xval - low)
    alfa = max.(zzz, xmin)
    zzz = upp - albefa .* (upp - xval)
    beta = min.(zzz, xmax)
    # Calculations of p0, q0, r0, P, Q, r and b.
    xmami = xmax - xmin
    xmamieps = 1.0e-5 .* eeen
    xmami = max.(xmami, xmamieps)
    xmamiinv = eeen ./ xmami
    ux1 = upp - xval
    ux2 = ux1 .* ux1
    xl1 = xval - low
    xl2 = xl1 .* xl1
    uxinv = eeen ./ ux1
    xlinv = eeen ./ xl1
    #
    p0 = copy(zeron)
    q0 = copy(zeron)
    p0 = max.( df0dx, 0.0)
    q0 = max.(-df0dx, 0.0)
    pq0 = p0 + q0
    p0 = p0 + 0.001 .* pq0
    q0 = q0 + 0.001 .* pq0
    p0 = p0 + raa0 .* xmamiinv
    q0 = q0 + raa0 .* xmamiinv
    p0 = p0 .* ux2
    q0 = q0 .* xl2
    r0 = f0val - transpose(p0) * uxinv - transpose(q0) * xlinv
    #
    P = zeros(Float64, m, n)
    Q = zeros(Float64, m, n)
    P = P .* transpose(ux2)
    Q = Q .* transpose(xl2)
    # b = P * uxinv + Q * xlinv - fval
    P = max.( dfdx, 0.0)
    Q = max.(-dfdx, 0.0)
    PQ = P+Q
    P = P + 0.001 .* PQ
    Q = Q + 0.001 .* PQ
    P = P + raa * transpose(xmamiinv)
    Q = Q + raa * transpose(xmamiinv)
    P = P .* transpose(ux2)
    Q = Q .* transpose(xl2)
    r = fval - P * uxinv - Q * xlinv
    b = -r
    # Solving the subproblem by a primal-dual Newton method
    xmma,ymma,zmma,lam,xsi,eta,mu,zet,s = subsolv(m,n,epsimin,low,upp,alfa,beta,p0,q0,P,Q,a0,a,b,c,d)
    # Calculations of f0app and fapp.
    ux1 = upp - xmma
    xl1 = xmma - low
    uxinv = eeen ./ ux1
    xlinv = eeen ./ xl1
    f0app = r0 + transpose(p0) * uxinv + transpose(q0) * xlinv
    fapp = r + P * uxinv + Q * xlinv
    # Return values
    return xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, f0app, fapp
end

# Function for solving the subproblem (can be used for MMA and GCMMA)
function subsolv(m::Int, n::Int, epsimin::Float64, low::Array{Float64,1}, upp::Array{Float64,1},
    alfa::Array{Float64,1}, beta::Array{Float64,1}, p0::Array{Float64,1}, q0::Array{Float64,1},
    P::Array{Float64,2}, Q::Array{Float64,2}, a0::Float64, a::Array{Float64,1}, b::Array{Float64,1},
    c::Array{Float64,1}, d::Array{Float64,1})

    # """
    # This function subsolv solves the MMA subproblem:
    #
    # minimize SUM[p0j/(uppj-xj) + q0j/(xj-lowj)] + a0*z + SUM[ci*yi + 0.5*di*(yi)^2],
    #
    # subject to SUM[pij/(uppj-xj) + qij/(xj-lowj)] - ai*z - yi <= bi,
    #     alfaj <=  xj <=  betaj,  yi >= 0,  z >= 0.
    #
    # Input:  m, n, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d.
    # Output: xmma,ymma,zmma, slack variables and Lagrange multiplers.
    # """

    een = ones(Float64, n)
    eem = ones(Float64, m)
    epsi = 1.0
    epsvecn = epsi .* een
    epsvecm = epsi .* eem
    x = 0.5 .* (alfa + beta)
    y = copy(eem)
    z = 1.0
    lam = copy(eem)
    xsi = een ./ (x - alfa)
    xsi = max.(xsi, een)
    eta = een ./ (beta - x)
    eta = max.(eta, een)
    mu = max.(eem, 0.5 .* c )
    zet = 1.0
    s = copy(eem)
    itera = 0
    while epsi > epsimin    # Start while epsi>epsimin
        epsvecn = epsi .* een
        epsvecm = epsi .* eem
        ux1 = upp - x
        xl1 = x - low
        ux2 = ux1 .* ux1
        xl2 = xl1 .* xl1
        uxinv1 = een ./ ux1
        xlinv1 = een ./ xl1
        plam = p0 + transpose(P) * lam
        qlam = q0 + transpose(Q) * lam
        gvec = P * uxinv1 + Q * xlinv1
        dpsidx = (plam ./ ux2) - (qlam ./ xl2)
        rex = dpsidx - xsi + eta
        rey = c + d .* y - mu - lam
        rez = a0 - zet - transpose(a) * lam
        relam = gvec - a .* z - y + s - b
        rexsi = xsi .* (x - alfa) - epsvecn
        reeta = eta .* (beta - x) - epsvecn
        remu = mu .* y - epsvecm
        rezet = zet * z - epsi
        res = lam .* s - epsvecm
        residu1 = [rex; rey; rez]
        residu2 = [relam; rexsi; reeta; remu; rezet; res]
        residu = [residu1; residu2]
        residunorm = norm(residu, 2)
        residumax = maximum(abs.( residu ))
        ittt = 0
        while (residumax > 0.9 * epsi) && (ittt < 200) # Start while (residumax>0.9*epsi) and (ittt<200)
            ittt = ittt + 1
            itera = itera + 1
            ux1 = upp - x
            xl1 = x - low
            ux2 = ux1 .* ux1
            xl2 = xl1 .* xl1
            ux3 = ux1 .* ux2
            xl3 = xl1 .* xl2
            uxinv1 = een ./ ux1
            xlinv1 = een ./ xl1
            uxinv2 = een ./ ux2
            xlinv2 = een ./ xl2
            plam = p0 + transpose(P) * lam
            qlam = q0 + transpose(Q) * lam
            gvec = P * uxinv1 + Q * xlinv1
            GG = P .* transpose(uxinv2) - Q .* transpose(xlinv2)
            dpsidx = (plam ./ ux2) - (qlam ./ xl2)
            delx = dpsidx - epsvecn ./ (x - alfa) + epsvecn ./ (beta - x)
            dely = c + d .* y - lam - epsvecm ./ y
            delz = a0 - transpose(a) * lam - epsi / z
            dellam = gvec - a .* z - y - b + epsvecm ./ lam
            diagx = plam ./ ux3 + qlam ./ xl3
            diagx = 2.0 .* diagx + xsi ./ (x - alfa) + eta ./ (beta - x)
            diagxinv = een ./ diagx
            diagy = d + mu ./ y
            diagyinv = eem ./ diagy
            diaglam = s ./ lam
            diaglamyi = diaglam + diagyinv
            if m < n # Start if m < n
                blam = dellam + dely ./ diagy - GG * (delx ./ diagx)
                bb = [blam; delz]
                Alam =  spdiagm(0 => diaglamyi) + (GG .* transpose(diagxinv)) * transpose(GG)
                AAr1 = [Alam a]
                AAr2 = transpose([a; -zet / z])
                AA = [AAr1; AAr2]
                solut = AA \ bb
                dlam = solut[1 : m]
                dz = solut[m+1]
                dx = -delx ./ diagx - (transpose(GG) * dlam) ./ diagx
            else
                diaglamyiinv = eem ./ diaglamyi
                dellamyi = dellam + dely ./ diagy
                Axx = spdiagm(0 => diagx) + (transpose(GG) .* transpose(diaglamyiinv)) * GG
                azz = zet / z + transpose(a) * (a ./ diaglamyi)
                axz = -transpose(GG) * (a ./ diaglamyi)
                bx = delx + transpose(GG) * (dellamyi ./ diaglamyi)
                bz = delz - transpose(a) * (dellamyi ./ diaglamyi)
                AAr1 = [Axx axz]
                AAr2 = [transpose(axz) azz]
                AA = [AAr1; AAr2]
                bb = [-bx; -bz]
                solut = AA \ bb
                dx = solut[1 : n]
                dz = solut[n+1]
                dlam = (GG * dx) ./ diaglamyi - dz .* (a ./ diaglamyi) + dellamyi ./ diaglamyi
            end # End if m<n
            dy = -dely ./ diagy + dlam ./ diagy
            dxsi = -xsi + epsvecn ./ (x - alfa) - (xsi .* dx) ./ (x - alfa)
            deta = -eta + epsvecn ./ (beta - x) + (eta .* dx) ./ (beta-x)
            dmu = -mu + epsvecm ./ y - (mu .* dy) ./ y
            dzet = -zet + epsi / z - zet * dz / z
            ds = -s + epsvecm ./ lam - (s .* dlam) ./ lam
            xx = [y; z; lam; xsi; eta; mu; zet; s]
            dxx = [dy; dz; dlam; dxsi; deta; dmu; dzet; ds]
            #
            stepxx = -1.01 .* dxx ./ xx
            stmxx = maximum(stepxx)
            stepalfa = -1.01 .* dx ./ (x - alfa)
            stmalfa = maximum(stepalfa)
            stepbeta = 1.01 .* dx ./ (beta - x)
            stmbeta = maximum(stepbeta)
            stmalbe = max(stmalfa, stmbeta)
            stmalbexx = max(stmalbe, stmxx)
            stminv = max(stmalbexx, 1.0)
            steg = 1.0 / stminv
            #
            xold = copy(x)
            yold = copy(y)
            zold = copy(z)
            lamold = copy(lam)
            xsiold = copy(xsi)
            etaold = copy(eta)
            muold = copy(mu)
            zetold = copy(zet)
            sold = copy(s)
            #
            itto = 0
            resinew = 2.0 * residunorm
            # Start: while (resinew>residunorm) and (itto<50)
            while (resinew > residunorm) && (itto < 50)
                itto = itto + 1
                x = xold + steg .* dx
                y = yold + steg .* dy
                z = zold + steg * dz
                lam = lamold + steg .* dlam
                xsi = xsiold + steg .* dxsi
                eta = etaold + steg .* deta
                mu = muold + steg .* dmu
                zet = zetold + steg * dzet
                s = sold + steg .* ds
                ux1 = upp - x
                xl1 = x - low
                ux2 = ux1 .* ux1
                xl2 = xl1 .* xl1
                uxinv1 = een ./ ux1
                xlinv1 = een ./ xl1
                plam = p0 + transpose(P) * lam
                qlam = q0 + transpose(Q) * lam
                gvec = P * uxinv1 + Q * xlinv1
                dpsidx = plam ./ ux2 - qlam ./ xl2
                rex = dpsidx - xsi + eta
                rey = c + d .* y - mu - lam
                rez = a0 - zet - transpose(a) * lam
                relam = gvec - a .* z - y + s - b
                rexsi = xsi .* (x - alfa) - epsvecn
                reeta = eta .* (beta - x) - epsvecn
                remu = mu .* y - epsvecm
                rezet = zet * z - epsi
                res = lam .* s - epsvecm
                residu1 = [rex; rey; rez]
                residu2 = [relam; rexsi; reeta; remu; rezet; res]
                residu = [residu1; residu2]
                resinew = norm(residu, 2)
                steg = steg / 2.0
            end # End: while (resinew>residunorm) and (itto<50)
            residunorm = copy(resinew)
            residumax = maximum(abs.(residu))
            steg = 2.0 * steg
        end # End: while (residumax>0.9*epsi) and (ittt<200)
        epsi = 0.1 * epsi
    end  # End: while epsi>epsimin
    xmma = copy(x)
    ymma = copy(y)
    zmma = copy(z)
    lamma = lam
    xsimma = xsi
    etamma = eta
    mumma = mu
    zetmma = zet
    smma = s
    # Return values
    return xmma, ymma, zmma, lamma, xsimma, etamma, mumma, zetmma, smma
end

# Function for Karush–Kuhn–Tucker check
function kktcheck(m::Int, n::Int, x::Array{Float64,1}, y::Array{Float64,1}, z::Float64,
    lam::Array{Float64,1}, xsi::Array{Float64,1}, eta::Array{Float64,1}, mu::Array{Float64,1},
    zet::Float64, s::Array{Float64,1}, xmin::Array{Float64,1}, xmax::Array{Float64,1},
    df0dx::Array{Float64,1}, fval::Array{Float64,1}, dfdx::Array{Float64,2}, a0::Float64,
    a::Array{Float64,1}, c::Array{Float64,1}, d::Array{Float64,1})

    # """
    # The left hand sides of the KKT conditions for the following nonlinear programming problem are
    # calculated.
    #
    # Minimize f_0(x) + a_0*z + sum(c_i*y_i + 0.5*d_i*(y_i)^2)
    # subject to  f_i(x) - a_i*z - y_i <= 0,   i = 1,...,m
    #             xmax_j <= x_j <= xmin_j,     j = 1,...,n
    #             z >= 0,   y_i >= 0,          i = 1,...,m
    #
    # INPUT:
    #
    #     m     = The number of general constraints.
    #     n     = The number of variables x_j.
    #     x     = Current values of the n variables x_j.
    #     y     = Current values of the m variables y_i.
    #     z     = Current value of the single variable z.
    #     lam   = Lagrange multipliers for the m general constraints.
    #     xsi   = Lagrange multipliers for the n constraints xmin_j - x_j <= 0.
    #     eta   = Lagrange multipliers for the n constraints x_j - xmax_j <= 0.
    #     mu    = Lagrange multipliers for the m constraints -y_i <= 0.
    #     zet   = Lagrange multiplier for the single constraint -z <= 0.
    #     s     = Slack variables for the m general constraints.
    #     xmin  = Lower bounds for the variables x_j.
    #     xmax  = Upper bounds for the variables x_j.
    #     df0dx = Vector with the derivatives of the objective function f_0
    #             with respect to the variables x_j, calculated at x.
    #     fval  = Vector with the values of the constraint functions f_i,
    #             calculated at x.
    #     dfdx  = (m x n)-matrix with the derivatives of the constraint functions
    #             f_i with respect to the variables x_j, calculated at x.
    #             dfdx(i,j) = the derivative of f_i with respect to x_j.
    #     a0    = The constants a_0 in the term a_0*z.
    #     a     = Vector with the constants a_i in the terms a_i*z.
    #     c     = Vector with the constants c_i in the terms c_i*y_i.
    #     d     = Vector with the constants d_i in the terms 0.5*d_i*(y_i)^2.
    #
    # OUTPUT:
    #
    #     residu     = the residual vector for the KKT conditions.
    #     residunorm = sqrt(residu'*residu).
    #     residumax  = max(abs(residu)).
    #
    # """

    rex = df0dx + transpose(dfdx) * lam - xsi + eta
    rey = c + d .* y - mu - lam
    rez = a0 - zet - transpose(a) * lam
    relam = fval - a * z - y + s
    rexsi = xsi .* (x - xmin)
    reeta = eta .* (xmax - x)
    remu = mu .* y
    rezet = zet * z
    res = lam .* s
    residu1 = [rex; rey; rez]
    residu2 = [relam; rexsi; reeta; remu; rezet; res]
    residu = [residu1; residu2]
    residunorm = norm(residu, 2)
    residumax = maximum(abs.(residu))
    return residu, residunorm, residumax
end

# Function for updating raa0 and raa
function raaupdate(xmma::Array{Float64,1}, xval::Array{Float64,1}, xmin::Array{Float64,1},
    xmax::Array{Float64,1}, low::Array{Float64,1}, upp::Array{Float64,1}, f0valnew::Float64,
    fvalnew::Array{Float64,1}, f0app::Float64, fapp::Array{Float64,1}, raa0::Float64,
    raa::Array{Float64,1}, raa0eps::Float64, raaeps::Array{Float64,1}, epsimin::Float64)

    # """
    # Values of the parameters raa0 and raa are updated during an inner iteration.
    # """

    raacofmin = 1e-12
    eeem = ones(Float64, size( raa))
    eeen = ones(Float64, size(xmma))
    xmami = xmax - xmin
    xmamieps = 0.00001 .* eeen
    xmami = max.(xmami, xmamieps)
    xxux = (xmma - xval) ./ (upp - xmma)
    xxxl = (xmma - xval) ./ (xmma - low)
    xxul = xxux .* xxxl
    ulxx = (upp - low) ./ xmami
    raacof = transpose(xxul) * ulxx
    raacof = max(raacof, raacofmin)
    #
    f0appe = f0app + 0.5 * epsimin
    if all(f0valnew .> f0appe)
        deltaraa0 = (1.0 / raacof) * (f0valnew - f0app)
        zz0 = 1.1 * (raa0 + deltaraa0)
        zz0 = min(zz0, 10.0 * raa0)
        raa0 = zz0
    end
    #
    fappe = fapp + (0.5 * epsimin) .* eeem;
    fdelta = fvalnew - fappe
    deltaraa = (fvalnew - fapp) ./ raacof
    zzz = 1.1 .* (raa + deltaraa)
    zzz = min.(zzz, 10.0 * raa)
    raa[findall(fdelta .> 0.0)] = zzz[findall(fdelta .> 0)]
    #
    return raa0, raa
end

# Function to check if the approsimations are conservative
function concheck(m::Int, epsimin::Float64, f0app::Float64, f0valnew::Float64,
    fapp::Array{Float64,1}, fvalnew::Array{Float64,1})

    # """
    # If the current approximations are conservative, the parameter conserv is set to 1.
    # """

    eeem = ones(Float64,m)
    f0appe = f0app + epsimin
    fappe = fapp + epsimin .* eeem
    arr1 = [f0appe; fappe]
    arr2 = [f0valnew; fvalnew]
    if all(arr1 .>= arr2)
        conserv = 1
    else
        conserv = 0
    end
    return conserv
end

# Calculate low, upp, raa0, raa in the beginning of each outer iteration
function asymp(outeriter::Int, n::Int, xval::Array{Float64,1}, xold1::Array{Float64,1},
    xold2::Array{Float64,1}, xmin::Array{Float64,1}, xmax::Array{Float64,1},
    low::Array{Float64,1}, upp::Array{Float64,1}, raa0::Float64, raa::Array{Float64,1},
    raa0eps::Float64, raaeps::Array{Float64,1}, df0dx::Array{Float64,1}, dfdx::Array{Float64,2})

    # """
    # Values on the parameters raa0, raa, low and upp are calculated in the beginning of each outer
    # iteration.
    # """

    eeen = ones(Float64, n)
    asyinit = 0.5
    asyincr = 1.2
    asydecr = 0.7
    xmami = xmax - xmin
    xmamieps = 0.00001 .* eeen
    xmami = max.(xmami, xmamieps)
    raa0 = transpose(abs.(df0dx)) * xmami
    raa0 = max(raa0eps, (0.1 / n) * raa0)
    raa = abs.(dfdx) * xmami
    raa = max.(raaeps, (0.1 ./ n) .* raa)
    if outeriter <= 2
        low = xval - asyinit .* xmami
        upp = xval + asyinit .* xmami
    else
        xxx = (xval - xold1) .* (xold1 - xold2)
        factor = copy(eeen)
        factor[findall(xxx .> 0)] .= asyincr
        factor[findall(xxx .< 0)] .= asydecr
        low = xval - factor .* (xold1 - low)
        upp = xval + factor .* (upp - xold1)
        lowmin = xval - 10.0 .* xmami
        lowmax = xval - 0.01 .* xmami
        uppmin = xval + 0.01 .* xmami
        uppmax = xval + 10.0 .* xmami
        low = max.(low, lowmin)
        low = min.(low, lowmax)
        upp = min.(upp, uppmax)
        upp = max.(upp, uppmin)
    end
    return low, upp, raa0, raa
end
