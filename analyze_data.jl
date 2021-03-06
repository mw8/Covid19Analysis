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
function smooth_iir_1(y, alpha=0.2)
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

# sum of derivatives of sigmoids (generalized logistic function)
function sigmoid_deriv_sum(a, t)
    @assert(length(a) % 4 == 0)
    b = reshape(a, 4, :)
    m = size(b, 2)
    y = zeros(length(t))
    for i = 1:m
        e = exp.((-10.0 * b[2,i]) * (t .- b[3,i]))
        y .+= (4.0 * b[1,i]) * e ./ ((1.0 .+ e).^(1.0 + b[4,i]))
    end
    y
end

# Jacobian of the weighted residual in a sum of sigmoids fit
function sigmoid_deriv_sum_jac(a, t, w)
    @assert(length(a) % 4 == 0)
    b = reshape(a, 4, :)
    m = size(b, 2)
    n = length(t)
    jac = zeros(n, 4, m)
    u = (4.0 / sqrt(n)) * w
    for i = 1:m
        e = exp.(-10.0 * b[2,i] * (t .- b[3,i]))
        d = (1 .+ e)
        g = b[1,i] * u .* e
        h = g .* d.^(-(2.0 + b[4,i])) .* (1.0 .- b[4,i] * e)
        jac[:,1,i] = u .* e ./ (d.^(1.0 + b[4,i]))
        jac[:,2,i] = h .* (-10.0 * (t .- b[3,i]))
        jac[:,3,i] = h * (10.0 * b[2,i])
        jac[:,4,i] = g .* d.^(-(1.0 + b[4,i])) .* (-log.(d))
    end
    reshape(jac, n, :)
end

# test for Jacobian of the residual in a sum of sigmoids fit
function test_sigmoid_deriv_sum_jac()
    n = 100
    t = collect(range(0, stop=1, length=n))
    w = ones(n)
    a = [1.5, 0.6, 0.8, 1, 2, 0.3, 1, 1.1]
    y0 = rand(n)
    # direct computation of Jacobian of residual
    jac0 = sigmoid_deriv_sum_jac(a, t, w)
    # finite difference approximation
    jac1 = zeros(n, 8)
    eps = 1e-6
    for i = 1:8
        a1 = copy(a)
        a1[i] -= eps
        res0 = w .* (sigmoid_deriv_sum(a1, t) - y0) / sqrt(n)
        a1[i] += 2 * eps
        res1 = w .* (sigmoid_deriv_sum(a1, t) - y0) / sqrt(n)
        jac1[:, i] = (res1 - res0) / (2 * eps)
    end
    println(maximum(abs.(jac0 - jac1)))
end

# fit a sigmoid sum using the Gauss-Newton method to optimize weighted least squares
function fit_sigmoid_deriv_sum_wls_gn(x_init, t, w, y0_raw, iter_max, debug::Bool=false)
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

    # set up initial values
    eps = 1.0
    eps_it = 0
    x1 = copy(x_init)
    res = w .* (sigmoid_deriv_sum(x1, t) - y0) / sqrt(n)
    jac = sigmoid_deriv_sum_jac(x1, t, w)
    err1 = norm(res)
    err_opt = err1

    # regularization coefficients
    x_init[1:4:end] .= 0.0
    c_reg = 1e-3 * zeros(n)
    c_reg[3:4:end] .= 0.0

    # Gauss-Newton iterations
    for it = 1:iter_max
        
        # update parameters
        x0 = copy(x1)
        x1 -= (jac' * jac + eps * I) \ (jac' * res) * 0.5 - c_reg .* (x1 - x_init)
        x1[1:4:end] = max.(min.(x1[1:4:end], 100.0), 0.01)
        x1[2:4:end] = max.(min.(x1[2:4:end],  10.0),  0.1)
        x1[3:4:end] = max.(min.(x1[3:4:end],   1.1), -0.1)
        x1[4:4:end] = max.(min.(x1[4:4:end],  10.0),  0.1)

        # update weighted residual and its Jacobian
        err0 = err1
        res = w .* (sigmoid_deriv_sum(x1, t) - y0) / sqrt(n)
        jac = sigmoid_deriv_sum_jac(x1, t, w)
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
    x_opt, err_opt / length(y0)
end

# fit sum of sigmoid functions (generalized logistics) using iteratively re-weighted least squares
function fit_sigmoid_deriv_sum(y0, n_ext, discount_last_5::Bool=true)
    n = length(y0)
    t0 = collect(range(0, stop=1, length=n))
    w0 = ones(n)
    if discount_last_5
        w0[end - 4:end] = [0.8, 0.5, 0.3, 0.2, 0.1]
    end
    opt_a1 = []
    opt_b1 = Inf
    for n_sig = 1:8
        a0 = ones(4, n_sig)
        a0[1,:] .= 0.3
        for i_sig = 1:n_sig
            a0[2,i_sig] = i_sig / (n_sig + 1.0)
        end
        a1, err = fit_sigmoid_deriv_sum_wls_gn(a0[:], t0, w0, y0, 300)
        b1 = 0.2 * n_sig + log(err) # bayesian information criterion
        if b1 < opt_b1
            opt_b1 = b1
            opt_a1 = a1
        end
    end
    t1 = [t0; 1.0 .+ (t0[2] - t0[1]) * collect(1:n_ext)]
    y1 = sigmoid_deriv_sum(opt_a1, t1)
    y1_max = sigmoid_deriv_sum(opt_a1, 10.0)
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

# hack to deal with twinx() undocumented/buggy behavior for dates: convert to int64
function dates_int64_xticks(t0)
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
    xt, x, x_offset
end

# plot total cases/deaths
function plot_tcd_county(cd, county, state, date, pop, set_ymax, n_days_ext=10, n_days_plot=100)
    # normalize by population (per 100,000)
    c0 = cd[:,1] / pop * 1e5
    d0 = cd[:,2] / pop * 1e5
    # get dates for each day
    n_days = size(cd, 1)
    t0 = collect(date - Day(n_days - 1):Day(1):date)
    # cutoff
    n_days_plot = min(n_days_plot, n_days)
    t0 = t0[end - (n_days_plot - 1):end]
    c0 = c0[end - (n_days_plot - 1):end]
    d0 = d0[end - (n_days_plot - 1):end]
    # plot cases and deaths
    if set_ymax
        c_ymax = 3200.0
        d_ymax = 150.0
    else
        c_ymax = 1.05 * maximum(c0)
        d_ymax = 2.05 * maximum(d0)
    end
    xt, x, x_offset = dates_int64_xticks(t0)
    p = plot(x, c0, color=:black, xticks=xt, xtickfontsize=8, ylabel="Total Cases (per 100k)", yguidefont=font(8),
             title="$county County, $state\n", titlefontsize=10, legend=false, ylims=(0, c_ymax))
    plot!(twinx(), d0, color=:red, grid=:off, xticks=false, legend=false,
          ylabel="Total Deaths (per 100k)", yguidefont=font(8, :red), ylims=(0, d_ymax))
    plot!(xformatter=x -> Dates.monthabbr(Date(Dates.UTD(x + x_offset))))
    p
end

# plot daily cases/deaths per day
function plot_dcd_county(cd, county, state, date, pop, set_ymax, n_days_ext=10, n_days_plot=100)
    # normalize by population (per 100,000)
    c0 = cd[:,1] / pop * 1e5
    d0 = cd[:,2] / pop * 1e5
    # fit sigmoid derivative to new cases
    c0s = smooth_iir_1(c0)
    d0s = smooth_iir_1(d0, 0.1)
    dc0 = diff(c0s)
    dd0 = diff(d0s)
    dc1, dc1_max = fit_sigmoid_deriv_sum(dc0, n_days_ext, true)
    dd1, dd1_max = fit_sigmoid_deriv_sum(dd0, n_days_ext, false)
    # get dates for each day
    n_days = size(cd, 1) - 1
    t0 = collect(date - Day(n_days - 1):Day(1):date)
    t1 = collect(date - Day(n_days - 1):Day(1):date + Day(n_days_ext))
    # cutoff
    n_days_plot = min(n_days_plot, n_days)
    t0 = t0[end - (n_days_plot - 1):end - 3]
    t1 = t1[end - (n_days_plot + n_days_ext - 1):end]
    dc0 = dc0[end - (n_days_plot - 1):end - 3]
    dd0 = dd0[end - (n_days_plot - 1):end - 3]
    dc1 = dc1[end - (n_days_plot + n_days_ext - 1):end]
    dd1 = dd1[end - (n_days_plot + n_days_ext - 1):end]
    # plot cases
    if set_ymax
        c_ymax = 40.0
        d_ymax = 1.0
    else
        c_ymax = max(1.05 * maximum(dc0), 40.0)
        d_ymax = max(2.05 * maximum(dd0), 1.0)
    end
    xt, x, x_offset = dates_int64_xticks(t0)
    p = plot(x, dc0, color=:black, xticks=false, ylabel="Daily Cases (per 100k)", yguidefont=font(8),
             title="$county County, $state\n", titlefontsize=10, legend=false, ylims=(0, c_ymax))
    tx = twinx()
    plot!(tx, dd0, color=:red, grid=:off, xticks=false, legend=false,
          ylabel="Daily Deaths (per 100k)", yguidefont=font(8, :red), ylims=(0, d_ymax))
    xt, x, x_offset = dates_int64_xticks(t1)
    plot!(x, dc1, color=:grey, xticks=xt, xtickfontsize=8, legend=false, ylims=(0, c_ymax))
    plot!(tx, dd1, color=:orange, grid=:off, xticks=false, legend=false, ylims=(0, d_ymax))
    plot!(xformatter=x -> Dates.monthabbr(Date(Dates.UTD(x + x_offset))))
    p
end

# plot cases/deaths over time and new cases versus total cases
function plot_county_list(df, county_list, set_ymax=false)
    n = length(county_list)
    date = df[end,:date]
    p = Array{Any,2}(undef, 2, n)
    for i = 1:n
        county, state = county_list[i]
        cd = load_cd_county(df, county, state)
        pop = load_pop_county(county, state)
        p[1,i] = plot_tcd_county(cd, county, state, date, pop, set_ymax)
        p[2,i] = plot_dcd_county(cd, county, state, date, pop, set_ymax)
    end
    plot(p..., layout=(n, 2), size=(800, 250 * n), margin=5mm, right_margin=15mm)
end


# load data
print("Loading Covid-19 county cases/deaths data...")
t = @elapsed df = CSV.read("covid-19-data/us-counties.csv", DataFrame)
@printf("finished in %.2f sec.\n", t)

# plot data
display(plot_county_list(df, [("Winnebago", "Illinois"),
                              ("Boone", "Illinois"),
                              ("DeKalb", "Illinois")]))

display(plot_county_list(df, [("Alameda", "California"),
                              ("Santa Clara", "California"),
                              ("Los Angeles", "California"),
                              ("San Diego", "California")], true))

display(plot_county_list(df, [("Santa Clara", "California"),
                              ("Los Angeles", "California"),
                              ("Broward", "Florida")]))
png("figures/example_2.png")