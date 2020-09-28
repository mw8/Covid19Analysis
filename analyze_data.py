import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# load population data for a county
def load_pop_county(county, state):
    c = pd.read_csv('county_pop.csv')
    return c[(c['county'] == county) & (c['state'] == state)].iloc[0]['population']


# load population data for a state
def load_pop_state(state):
    c = pd.read_csv('state_pop.csv')
    return c[c['state'] == state].iloc[0]['population']


# load case and death data for all states
def load_cd_states():
    c = pd.read_csv('covid-19-data/us-states.csv')
    date = c['date'][len(c) - 1]
    states = c.state.unique()
    n_states = len(states)
    cd = []
    for i in range(n_states):
        r = c[c['state'] == states[i]]
        cd.append(np.array(r[['cases', 'deaths']]))
    return states, cd, date


# load case and death data for a county
def load_cd_county(csv_data, county, state):
    r = csv_data[(csv_data['county'] == county) & (csv_data['state'] == state)]
    return np.array(r[['cases', 'deaths']])


# apply first-order IIR to smooth data
def smooth_iir_1(y):
    alpha = 0.2
    n = len(y)
    # first order IIR forward scan
    z0 = y.copy()
    z = y[0]
    for i in range(n):
        z = alpha * y[i] + (1 - alpha) * z
        z0[i] = z
    # first order IIR backward scan
    z1 = y.copy()
    z = y[-1]
    for i in range(n - 1, -1, -1):
        z = alpha * y[i] + (1 - alpha) * z
        z1[i] = z
    # return average
    return (z0 + z1) * 0.5


# sum of sigmoids (generalized logistic functions)
def sigmoid_sum(a, t):
    assert (len(a) % 4 == 0)
    m = len(a) // 4
    ss = 0
    for i in range(m):
        ss += a[4 * i] / ((1.0 + np.exp(-10.0 * a[4 * i + 1] * (t - a[4 * i + 2]))) ** a[4 * i + 3])
    return ss


# compute the Jacobian of the residual in a sum of sigmoids fit
def sigmoid_sum_jac(a, t, w):
    assert (len(a) % 4 == 0)
    m = len(a) // 4
    n = t.shape[0]
    jac = np.zeros((n, len(a)))
    for i in range(m):
        e = np.exp(-10.0 * a[4 * i + 1] * (t - a[4 * i + 2]))
        d = (1.0 + e)
        dp = w * d ** (-a[4 * i + 3]) / np.sqrt(n)
        dpd = w * a[4 * i + 0] * (-a[4 * i + 3]) * (d ** (-a[4 * i + 3] - 1)) / np.sqrt(n)
        jac[:, 4 * i + 0] = dp
        jac[:, 4 * i + 1] = dpd * (-10.0 * (t - a[4 * i + 2])) * e
        jac[:, 4 * i + 2] = dpd * 10.0 * a[4 * i + 1] * e
        jac[:, 4 * i + 3] = a[4 * i + 0] * -dp * np.log(d)
    return jac


def sigmoid_sum_jac_test():
    n = 100
    t = np.linspace(0., 1., n)
    w = np.ones(n)
    a = np.array([1.5, 0.6, 0.8, 1, 2, 0.3, 1, 1.1])
    y0 = np.random.rand(n)
    # direct computation of jacobian of residual
    jac0 = sigmoid_sum_jac(a, t, w)
    # finite difference approximation
    jac1 = np.zeros((n, 8))
    eps = 1e-6
    for i in range(8):
        a1 = a.copy()
        a1[i] -= eps
        res0 = w * (sigmoid_sum(a1, t) - y0) / np.sqrt(n)
        a1[i] += 2 * eps
        res1 = w * (sigmoid_sum(a1, t) - y0) / np.sqrt(n)
        jac1[:, i] = (res1 - res0) / (2 * eps)
    print((jac0 - jac1).max(0))


# fit a sigmoid sum using the Gauss-Newton method to optimize weighted least squares
def fit_sigmoid_sum_wls_gn(x_init, t, w, y0_raw, iter_max, debug=False):
    # initialize return value
    x_opt = x_init.copy()

    # normalize data to fit
    n = x_init.shape[0]
    y0_max = np.max(y0_raw)
    y0 = y0_raw / y0_max

    # Levenberg-Marquardt regularization parameters
    eps_min = 1e-6
    eps_max = 1e+2
    eps_inc = 3
    eps_dec = 0.5

    # set up initial values
    eps = 1
    eps_it = 0
    x1 = x_init.copy()
    res = w * (sigmoid_sum(x1, t) - y0) / np.sqrt(n)
    jac = sigmoid_sum_jac(x1, t, w)
    err1 = np.linalg.norm(res)
    err_opt = err1

    # Gauss-Newton iterations
    for it in range(iter_max):

        # update parameters
        x0 = x1.copy()
        x1 -= np.linalg.solve(jac.T @ jac + eps * np.eye(n), jac.T @ res) * 0.5
        x1[0::4] = np.maximum(np.minimum(x1[0::4], 100), 0.01)
        x1[1::4] = np.maximum(np.minimum(x1[1::4], 5), 0.1)
        x1[2::4] = np.maximum(np.minimum(x1[2::4], 2),-1.0)
        x1[3::4] = np.maximum(np.minimum(x1[3::4], 5), 0.1)

        # update weighted residual and its Jacobian
        err0 = err1
        res = w * (sigmoid_sum(x1, t) - y0) / np.sqrt(n)
        jac = sigmoid_sum_jac(x1, t, w)
        err1 = np.linalg.norm(res)

        # lock in parameters if the error goes down
        if err1 < err_opt:
            err_opt = err1
            x_opt = x1.copy()

        # update epsilon based on the error going up or down
        if err1 <= err0 + 1e-2:
            eps *= eps_dec
            if eps < eps_min:
                eps = eps_min
            if eps == eps_min:
                if eps_it == 3:
                    break
                else:
                    eps_it += 1
        else:  # backtrack
            if eps == eps_max:
                if eps_it == 3:
                    break
                else:
                    eps_it += 1
            else:
                eps_it = 0
                err1 = err0
                x1 = x0.copy()
                eps *= eps_inc
                if eps > eps_max:
                    eps = eps_max

        # possibly output debug information
        if debug:
            print(f'Iter: {it:2d},    Eps: {eps:.6f},    Err: {err1:.6f},    Best Err: {err_opt:.6f}')
    if debug:
        print()
    x_opt[::4] *= y0_max
    return x_opt, err_opt


# fit sum of sigmoid functions (generalized logistics) using iteratively re-weighted least squares
def fit_sigmoid_sum(y0, n_ext, discount_last_5):
    n = len(y0)
    t0 = np.linspace(0., 1., n)
    w0 = np.ones(n)
    if discount_last_5:
        w0[-5:] = [0.8, 0.5, 0.3, 0.2, 0.1]
    opt_a1 = []
    opt_b1 = float('inf')
    for n_sig in range(1, 5):
        a0 = np.ones(4 * n_sig)
        for i_sig in range(0, n_sig):
            a0[4 * i_sig + 2] = (i_sig + 1.0) / (n_sig + 1.0)
        a1, err = fit_sigmoid_sum_wls_gn(a0, t0, w0, y0, 200)
        b1 = 10.0 * math.log(err / n) + n_sig * math.log(n) # bayesian information criterion
        if b1 < opt_b1:
            opt_b1 = b1
            opt_a1 = a1
    t1 = np.concatenate((t0, 1. + (t0[1] - t0[0]) * (np.arange(n_ext) + 1.)))
    y1 = sigmoid_sum(opt_a1, t1)
    y1_max = sigmoid_sum(opt_a1, 10.)
    return y1, y1_max


# estimate the linear growth rate (derivative) of a time series by linear fit for every consecutive 7 days
def est_growth_rate(y):
    m = 7
    n = y.shape[0]
    gr = np.zeros(n - m + 1)
    x = np.arange(m)
    for i in range(n - m + 1):
        gr[i] = np.polyfit(x, y[i:i + m], 1)[0]
    return gr


# plot cases and deaths over time
def plot_cd_county(ax1, cd, county, state, date, pop, n_days_ext, n_days_plot):
    # number of days to plot and extrapolate
    n_days = cd.shape[0]
    n_days_plot = min(n_days_plot, n_days)

    # normalize by population (cases/deaths per 100,000)
    c0 = cd[:, 0] / pop * 1e5
    d0 = cd[:, 1] / pop * 1e5

    # fit sigmoid to cases and deaths
    t0 = np.flip(np.arange(n_days) + 1.)
    t1 = np.flip(np.arange(-n_days_ext, n_days) + 1.)
    c1, c1_max = fit_sigmoid_sum(smooth_iir_1(c0), n_days_ext, True)
    d1, d1_max = fit_sigmoid_sum(smooth_iir_1(d0), n_days_ext, False)

    # get datetimes for each day
    dt_now = np.datetime64(date)
    to_days = lambda t: dt_now - np.timedelta64(int(t) - 1, 'D')
    dt0 = list(map(to_days, t0))
    dt1 = list(map(to_days, t1))

    # title
    ax1.set_title('{} County, {}'.format(county, state), {'fontsize': 10})

    # plot cases asymptote
    ax1.plot([dt1[0], dt1[-1]], [c1_max, c1_max], lw=0.5, color='tab:gray')

    # plot deaths asymptote
    ax2 = ax1.twinx()
    ax2.plot([dt1[0], dt1[-1]], [d1_max, d1_max], lw=0.5, color='xkcd:baby pink')

    # plot cases
    ax1.plot(dt1, c1, color='tab:gray')
    ax1.plot(dt0, c0, color='k')
    ax1.set_ylabel('Total Cases per 100k')
    ax1.set_xlim(dt0[-n_days_plot], dt1[-1])
    ax1.set_ylim(0, 1.25 * np.max(c0))

    # plot deaths
    ax2.plot(dt1, d1, color='xkcd:baby pink')
    ax2.plot(dt0, d0, color='tab:red')
    ax2.set_ylabel('Total Deaths per 100k', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_ylim(0, 2.5 * np.max(d0))

    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))


# plot new cases versus total cases over time
def plot_nc_county(ax, cd, county, state, date, pop, n_days_ext, n_days_plot):
    # number of days to plot and extrapolate
    n_days = cd.shape[0]
    n_days_plot = min(n_days_plot, n_days)
    c0 = cd[:, 0] / pop * 1e5
    c0s = smooth_iir_1(c0)

    # fit sigmoid to cases
    t0 = np.flip(np.arange(n_days) + 1.)
    t1 = np.flip(np.arange(-n_days_ext, n_days) + 1.)
    c1, c1_max = fit_sigmoid_sum(c0s, n_days_ext, True)

    # get new cases per day
    n_days = n_days - 1
    t0 = t0[1:]
    t1 = t1[1:]
    dc0 = np.diff(c0s)
    dc1 = np.diff(c1)

    # cutoff
    n_days_plot = min(n_days_plot, n_days)
    t0 = t0[-n_days_plot:]
    t1 = t1[-n_days_plot - n_days_ext:]
    dc0 = dc0[-n_days_plot:]
    dc1 = dc1[-n_days_plot - n_days_ext:]

    # get datetimes for each day
    dt_now = np.datetime64(date)
    to_days = lambda t: dt_now - np.timedelta64(int(t) - 1, 'D')
    dt0 = list(map(to_days, t0))
    dt1 = list(map(to_days, t1))

    # plot cases
    ax.set_title('{} County, {}'.format(county, state), {'fontsize': 10})
    ax.plot(dt0[:-3], dc0[:-3], color='tab:gray', marker='.')
    ax.plot(dt1, dc1, color='k')
    ax.set_xlim(dt1[0], dt1[-1])
    ax.set_ylabel('New Cases per 100k\nper Day (smoothed)')

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))


# plot cases/deaths over time and new cases versus total cases
def plot_county_list(county_list):
    n = len(county_list)
    csv_data = pd.read_csv('covid-19-data/us-counties.csv')
    date = csv_data['date'][len(csv_data) - 1]
    fig, axs = plt.subplots(2, n, figsize=(16, 6))
    plt.subplots_adjust(wspace=0.45, hspace=0.3)
    for i in range(n):
        county, state = county_list[i]
        cd = load_cd_county(csv_data, county, state)
        pop = load_pop_county(county, state)
        plot_cd_county(axs[0, i], cd, county, state, date, pop, 10, 200)
        plot_nc_county(axs[1, i], cd, county, state, date, pop, 10, 200)


# table of new deaths per day for the top 10 states
def plot_nd_states():
    n_days = 50
    states, cd_states, date_str = load_cd_states()
    n_states = len(states)
    # compute growth rates and their corresponding scores
    gr = []
    gr_score = np.zeros(n_states)
    for i in range(n_states):
        pop = load_pop_state(states[i])
        gr.append(est_growth_rate(smooth_iir_1(cd_states[i][-n_days - 7:, 0])) / pop * 1e5)
        gr_score[i] = (np.mean(gr[i][-14:]) + (np.max(gr[i]) - np.min(gr[i])) / 10) * (pop > 2e5)
    # sort based on recent growth rate
    idx = np.argsort(gr_score)
    gr = [gr[i] for i in idx]
    st = [states[i] for i in idx]
    # plot worst states
    line_styles = ['r-', 'r--', 'r-.', 'r:', 'k-', 'k--', 'k-.', 'k:', 'b-', 'b--', 'b-.', 'b:']
    n_top = 8
    fig = plt.figure(figsize=(8, 5))
    x = np.arange(n_days, 0, -1)
    curve_idx = 0
    for i in range(n_top - 1, -1, -1):
        n = min(len(gr[n_states - n_top + i]), n_days)
        plt.plot(x[-n:], gr[n_states - n_top + i][-n:], line_styles[curve_idx], label=st[n_states - n_top + i], lw=2.0)
        curve_idx += 1
    # add specific states
    plot_states = ['California']
    for i in range(len(plot_states)):
        j = st.index(plot_states[i])
        if j >= n_top:
            n = min(len(gr[j]), n_days)
            plt.plot(x[-n:], gr[j][-n:], line_styles[curve_idx], label=st[j], lw=2.0)
            curve_idx += 1
    plt.xlabel('Days Ago')
    plt.ylabel('New Cases per 100k (Weekly Average)')
    plt.xlim(n_days, 4)
    plt.legend(loc='upper left', fontsize='small')
    plt.title(date_str)


if __name__ == "__main__":
    plot_county_list([('Santa Clara', 'California'), ('Los Angeles', 'California'), ('Broward', 'Florida')])
    plot_county_list([('Winnebago', 'Illinois'), ('Cook', 'Illinois'), ('DuPage', 'Illinois')])
    # plot_nd_states()
    plt.show()
