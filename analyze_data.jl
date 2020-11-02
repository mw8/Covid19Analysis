using Printf, DataFrames, CSV, LinearAlgebra, Plots, Plots.PlotMeasures, Dates
pyplot()

# load population data for a county
function load_pop_county(county::String, state::String)
    df = CSV.read("county_pop.csv", DataFrame)
    df[(df.county .== county) .& (df.state .== state), :population][1]
end

# load population data for a state
function load_pop_state(state::String)
    df = CSV.read("state_pop.csv", DataFrame)
    df[df.state .== state, :population][1]
end

# load case and death data for a single county from a dataframe
function load_cd_county(df::DataFrame, county::String, state::String)
    convert(Array{Int64,2}, df[(df.county .== county) .& (df.state .== state), [:cases, :deaths]])
end

# load case and death data for all states
function load_cd_states()
    df = CSV.read("covid-19-data/us-states.csv", DataFrame)
    date = df.date[end]
    states = unique(df.state)
    n_states = length(states) # not 50, includes territories, etc
    cd_data = Array{Any,1}(undef, n_states)
    for i in 1:n_states
        cd_data[i] = convert(Array{Int64,2}, df[df.state .== states[i], [:cases, :deaths]])
    end
    states, cd_data, date
end

# apply bidirectional first-order IIR to smooth data
function smooth_iir_1(y)
    alpha = 0.2
    n = length(y)
    # first order IIR forward
    z0 = copy(y)
    z = y[1]
    for i in 1:n
        z = alpha * y[i] + (1 - alpha) * z
        z0[i] = z
    end
    # first order IIR backward
    z1 = copy(y)
    z = y[end]
    for i = n:-1:1
        z = alpha * y[i] + (1 - alpha) * z
        z1[i] = z
    end
    (z0 + z1) * 0.5
end

# sum of sigmoids (generalized logistic function)
function sigmoid_sum(a, t)
    @assert(length(a) % 4 == 0)
    b = reshape(a, 4, :)
    m = size(b, 2)
    y = zeros(length(t))
    for i = 1:m
        y .+= b[1,i] ./ ((1.0 .+ exp.((-10.0 * b[2,i]) * (t .- b[3,i]))).^b[4,i])
    end
    y
end

# Jacobian of the residual in a sum of sigmoids fit
function sigmoid_sum_jac(a, t, w)
    @assert(length(a) % 4 == 0)
    b = reshape(a, 4, :)
    m = size(b, 2)
    n = length(t)
    jac = zeros(n, 4, m)
    for i = 1:m
        e = exp.(-10.0 * b[2,i] * (t .- b[3,i]))                    # exponential term
        d = (1 .+ e)                                                # denominator
        p = w .* (d.^(-b[4,i])) ./ sqrt(n)                          # weighted denom to a power
        v = w .* (b[1,i] * -b[4,i]) .* (d.^(-b[4,i] - 1)) / sqrt(n) # derivative of p
        jac[:,1,i] = p
        jac[:,2,i] = v .* (-10.0 * (t .- b[3,i])) .* e
        jac[:,3,i] = v .* (10.0 * b[2,i]) .* e
        jac[:,4,i] = b[1,i] * -p .* log.(d)
    end
    reshape(jac, n, :)
end

# test for Jacobian of the residual in a sum of sigmoids fit
function test_sigmoid_sum_jac()
    n = 100
    t = collect(range(0, stop=1, length=n))
    w = ones(n)
    a = [1.5, 0.6, 0.8, 1, 2, 0.3, 1, 1.1]
    y0 = rand(n)
    # direct computation of Jacobian of residual
    jac0 = sigmoid_sum_jac(a, t, w)
    # finite difference approximation
    jac1 = zeros(n, 8)
    eps = 1e-6
    for i = 1:8
        a1 = copy(a)
        a1[i] -= eps
        res0 = w .* (sigmoid_sum(a1, t) - y0) / sqrt(n)
        a1[i] += 2 * eps
        res1 = w .* (sigmoid_sum(a1, t) - y0) / sqrt(n)
        jac1[:, i] = (res1 - res0) / (2 * eps)
    end
    println(maximum(abs.(jac0 - jac1)))
end

# fit a sigmoid sum using the Gauss-Newton method to optimize weighted least squares
function fit_sigmoid_sum_wls_gn(x_init, t, w, y0_raw, iter_max, debug::Bool=false)
    # initialize return value
    x_opt = copy(x_init)

    # normalize data to fit
    n = length(x_init)
    y0_max = maximum(y0_raw)
    y0 = y0_raw / y0_max

    # Levenberg-Marquardt parameters
    eps_min = 1e-6
    eps_max = 1e+2
    eps_inc = 3.0
    eps_dec = 0.5

    # regularization coefficients
    c_reg = zeros(n)
    c_reg[1:4:end] .= 1e-3

    # set up initial values
    eps = 1.0
    eps_it = 0
    x1 = copy(x_init)
    res = w .* (sigmoid_sum(x1, t) - y0) / sqrt(n)
    jac = sigmoid_sum_jac(x1, t, w)
    err1 = norm(res)
    err_opt = err1

    # Gauss-Newton iterations
    for it = 1:iter_max
        
        # update parameters
        x0 = copy(x1)
        x1 -= (jac' * jac + eps * I) \ (jac' * res) * 0.5 - (c_reg .* x1)
        x1[1:4:end] = max.(min.(x1[1:4:end], 100.0), 0.01)
        x1[2:4:end] = max.(min.(x1[2:4:end],  10.0),  0.1)
        x1[3:4:end] = max.(min.(x1[3:4:end],   1.5), -0.5)
        x1[4:4:end] = max.(min.(x1[4:4:end],  10.0),  0.1)

        # update weighted residual and its Jacobian
        err0 = err1
        res = w .* (sigmoid_sum(x1, t) - y0) / sqrt(n)
        jac = sigmoid_sum_jac(x1, t, w)
        err1 = norm(res)

        # lock in parameters if the error goes down
        if err1 < err_opt
            err_opt = err1
            x_opt = copy(x1)
        end

        # update epsilon based on the error going up or down
        if err1 <= err0 + 1e-2
            eps *= eps_dec
            if eps < eps_min
                eps = eps_min
            end
            if eps == eps_min
                if eps_it == 3
                    break
                else
                    eps_it += 1
                end
            end
        else # backtrack
            if eps == eps_max
                if eps_it == 3
                    break
                else
                    eps_it += 1
                end
            else
                eps_it = 0
                err1 = err0
                x1 = copy(x0)
                eps *= eps_inc
                if eps > eps_max
                    eps = eps_max
                end
            end
        end

        # possibly output debug information
        if debug
            @printf("Iter: %2d,    Eps: %.6f,    Err: %.6f,    Best Err: %.6f\n",
                    it, eps, err1, err_opt)
        end
    end
    if debug
        println()
    end

    x_opt[1:4:end] *= y0_max
    x_opt, err_opt
end

# fit sum of sigmoid functions (generalized logistics) using iteratively re-weighted least squares
function fit_sigmoid_sum(y0, n_ext, discount_last_5::Bool=true)
    n = length(y0)
    t0 = collect(range(0, stop=1, length=n))
    w0 = ones(n)
    if discount_last_5
        w0[end - 4:end] = [0.8, 0.5, 0.3, 0.2, 0.1]
    end
    opt_a1 = []
    opt_b1 = Inf
    for n_sig = 1:4
        a0 = ones(4, n_sig)
        for i_sig = 1:n_sig
            a0[2,i_sig] = (i_sig + 1.0) / (n_sig + 1.0)
        end
        a1, err = fit_sigmoid_sum_wls_gn(a0[:], t0, w0, y0, 200)
        b1 = 0.5 * n_sig + log(err) # bayesian information criterion
        if b1 < opt_b1
            opt_b1 = b1
            opt_a1 = a1
        end
    end
    t1 = [t0; 1.0 .+ (t0[2] - t0[1]) * collect(1:n_ext)]
    y1 = sigmoid_sum(opt_a1, t1)
    y1_max = sigmoid_sum(opt_a1, 10.0)
    y1, y1_max
end

# estimate the linear growth rate (derivative) of a time series by linear fit for every consecutive 7 days
function est_growth_rate(y)
    m = 7
    n = length(y)
    g = zeros(n - m + 1)
    h = floor(m / 2)
    X = hcat(ones(m), collect(-h:m - 1 - h))
    for i = 1:n - m + 1
        g[i] = (X \ y[i:i + m - 1])[2]
    end
    g
end

function filter_first_of_month(t0)
    # find unique first days of each month
    t1 = unique(Dates.firstdayofmonth.(t0))
    if t1[1] < t0[1]
        t1 = t1[2:end]
    end
    t1
end

# plot covid19 cases and deaths over time
function plot_cd_county(cd, county, state, date, pop, n_days_ext=10, n_days_plot=250)
    # normalize by population (cases/deaths per 100,000)
    c0 = cd[:,1] / pop * 1e5
    d0 = cd[:,2] / pop * 1e5
    # fit sigmoid to cases and deaths
    c1, c1_max = fit_sigmoid_sum(smooth_iir_1(c0), n_days_ext, true)
    d1, d1_max = fit_sigmoid_sum(smooth_iir_1(d0), n_days_ext, false)
    # get dates for each day
    n_days = size(cd, 1)
    t0 = collect(date - Day(n_days - 1):Day(1):date)
    # cutoff
    n_days_plot = min(n_days_plot, n_days)
    t0 = t0[end - (n_days_plot - 1):end]
    c0 = c0[end - (n_days_plot - 1):end]
    d0 = d0[end - (n_days_plot - 1):end]
    # hack to deal with x ticks for two y axes (twinx() is undocumented/buggy)
    x = collect(0:length(t0) - 1)
    x_offset = Dates.value(t0[1])
    tt = filter_first_of_month(t0)
    xt = zeros(length(tt))
    j = 1
    for i = 1:length(t0)
        if tt[j] == t0[i]
            xt[j] = x[i]
            j += 1
            if j > length(tt)
                break
            end
        end
    end
    # plot cases and deaths
    p = plot(x, c0, color=:black, xticks=xt, xtickfontsize=8, ylabel="Cases (per 100k)",
             title="$county County, $state - $date\n", titlefontsize=12, legend=false, ylims=(0, 1.1 * maximum(c0)))
    plot!(twinx(), d0, color=:red, grid=:off, xticks=false, legend=false,
          ylabel="Deaths (per 100k)", yguidefont=font(12, :red), ylims=(0, 2 * maximum(d0)))
    plot!(xformatter=x -> Dates.monthabbr(Date(Dates.UTD(x + x_offset))))
    p
end

# plot new cases over time
function plot_nc_county(cd, county, state, date, pop, n_days_ext=10, n_days_plot=250)
    # normalize by population (cases per 100,000)
    c0 = cd[:,1] / pop * 1e5
    # fit sigmoid to cases
    c0s = smooth_iir_1(c0)
    c1, c1_max = fit_sigmoid_sum(c0s, n_days_ext)
    # get new cases per day
    dc0 = diff(c0s)
    dc1 = diff(c1)
    # get dates for each day
    n_days = size(cd, 1) - 1
    t0 = collect(date - Day(n_days - 1):Day(1):date)
    t1 = collect(date - Day(n_days - 1):Day(1):date + Day(n_days_ext))
    # cutoff
    n_days_plot = min(n_days_plot, n_days)
    t0 = t0[end - (n_days_plot - 1):end]
    t1 = t1[end - (n_days_plot + n_days_ext - 1):end]
    dc0 = dc0[end - (n_days_plot - 1):end]
    dc1 = dc1[end - (n_days_plot + n_days_ext - 1):end]
    # plot cases
    xt = filter_first_of_month(t1)
    p = plot(t0, dc0, color=:gray, legend=false);
    plot!(t1, dc1, xticks=xt, color=:black, xtickfontsize=8, ylabel="New Cases (per 100k)", legend=false)
    plot!(xformatter=x -> Dates.monthabbr(Date(Dates.UTD(x))))
    p
end

# plot cases/deaths over time and new cases versus total cases
function plot_county_list(df, county_list)
    n = length(county_list)
    date = df[end,:date]
    p = Array{Any,2}(undef, n, 2)
    for i = 1:n
        county, state = county_list[i]
        cd = load_cd_county(df, county, state)
        pop = load_pop_county(county, state)
        p[i,1] = plot_cd_county(cd, county, state, date, pop)
        p[i,2] = plot_nc_county(cd, county, state, date, pop)
    end
    plot(p..., layout=(2, n), size=(1200, 600), margin=10mm, right_margin=15mm)
end

# load data
print("Loading Covid-19 county cases/deaths data...")
t = @elapsed df = CSV.read("covid-19-data/us-counties.csv", DataFrame)
@printf("finished in %.2f sec.\n", t)

# plot data
display(plot_county_list(df, [("Winnebago", "Illinois"),
                              ("Cook", "Illinois"),
                              ("DuPage", "Illinois")]))

display(plot_county_list(df, [("Santa Clara", "California"),
                              ("Los Angeles", "California"),
                              ("Broward", "Florida")]))
