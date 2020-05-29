import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
        r = c[c['state']==states[i]]
        cd.append(np.array(r[['cases', 'deaths']]))
    return states, cd, date


# load case and death data for a county
def load_cd_county(csv_data, county, state):
    r = csv_data[(csv_data['county'] == county) & (csv_data['state'] == state)]
    return np.array(r[['cases', 'deaths']])


# apply first-order IIR to smooth data
def smooth_iir_1(y):
    alpha = 0.3
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


# generalized logistic function (sigmoid)
def sigmoid(a, t):
    return a[0] / ((a[1] + np.exp(-a[2] * (t - a[3]))) ** a[4])


# compute the Jacobian of the residual in a generalized logistic fit
def sigmoid_jac(a, t, w):
    n = t.shape[0]
    jac = np.zeros((n, 5))
    e = np.exp(-a[2] * (t - a[3]))
    d = (a[1] + e)
    dp = w * d ** (-a[4]) / np.sqrt(n)
    dpd = w * a[0] * (-a[4]) * (d ** (-a[4] - 1)) / np.sqrt(n)
    jac[:, 0] = dp
    jac[:, 1] = dpd
    jac[:, 2] = dpd * (-(t - a[3])) * e
    jac[:, 3] = dpd * a[2] * e
    jac[:, 4] = a[0] * -dp * np.log(d)
    return jac


def sigmoid_jac_test():
    n = 100
    t = np.linspace(0., 1., n)
    w = np.ones(n)
    a = np.array([1.5, 1., 6., 0.8, 1.])
    y0 = np.random.rand(n)
    # direct computation of jacobian of residual
    jac0 = sigmoid_jac(a, t, w)
    # finite difference approximation
    jac1 = np.zeros((n, 5))
    eps = 1e-6
    res0 = w * (sigmoid(a, t) - y0) / np.sqrt(n)
    for i in range(5):
        a1 = a.copy()
        a1[i] += eps
        res1 = w * (sigmoid(a1, t) - y0) / np.sqrt(n)
        jac1[:, i] = (res1 - res0) / eps
    print((jac0 - jac1).max(0))


# fit a sigmoid using the Gauss-Newton method to optimize weighted least squares
def fit_sigmoid_wls_gn(x_init, t, w, y0_raw, iter_max, debug=False):
    # initialize return value
    x_opt = x_init.copy()

    # normalize data to fit
    n = x_init.shape[0]
    y0_max = np.max(y0_raw)
    y0 = y0_raw / y0_max

    # Levenberg-Marquardt regularization parameters
    tol = 1e-3
    eps_min = 1e-5
    eps_max = 1e+3
    eps_inc = 3
    eps_dec = 0.66

    # set up initial values
    eps = 0.01
    eps_it = 0
    x1 = x_init.copy()
    res = w * (sigmoid(x1, t) - y0) / np.sqrt(n)
    jac = sigmoid_jac(x1, t, w)
    err1 = np.linalg.norm(res)
    err_opt = err1

    # Gauss-Newton iterations
    for it in range(iter_max):

        # update parameters
        x0 = x1.copy()
        x1 -= np.linalg.solve(jac.T @ jac + eps * np.eye(n), jac.T @ res) * 0.5
        x1 = np.maximum(x1, 1e-2)

        # update weighted residual and its Jacobian
        err0 = err1
        res = w * (sigmoid(x1, t) - y0) / np.sqrt(n)
        jac = sigmoid_jac(x1, t, w)
        err1 = np.linalg.norm(res)

        # lock in parameters if the error goes down
        if err1 < err_opt:
            err_opt = err1
            x_opt = x1.copy()

        # update epsilon based on the error going up or down
        if err1 <= err0 + tol:
            eps *= eps_dec
            if eps < eps_min:
                eps = eps_min
            if eps == eps_min:
                if eps_it == 4:
                    break
                else:
                    eps_it += 1
        else:  # backtrack
            eps_it = 0
            err1 = err0
            x1 = x0.copy()
            eps *= eps_inc
            if eps > eps_max:
                eps = eps_max

        # possibly output debug information
        if debug:
            print(f'Iter: {it:2d},    Eps: {eps:.5f},    Err: {err1:.5f},    Best Err: {err_opt:.5f}')
    if debug:
        print()
    x_opt[0] *= y0_max
    return x_opt


# fit sigmoid function (generalized logistic) using iteratively reweighted least squares
def fit_sigmoid(y0, n_ext):
    n = len(y0)
    t0 = np.linspace(0., 1., n)
    w0 = np.ones(n)
    a0 = np.array([1.5, 1, 5, 0.75, 1])
    a1 = fit_sigmoid_wls_gn(a0, t0, w0, y0, 200)
    t1 = np.concatenate((t0, 1. + (t0[1] - t0[0]) * (np.arange(n_ext) + 1.)))
    y1 = sigmoid(a1, t1)
    y1_max = sigmoid(a1, 100.)
    return y1, y1_max


# estimate the linear growth rate (derivative) of a time series by linear fit for every consecutive 7 days
def est_growth_rate(y):
    n = y.shape[0]
    gr = np.zeros(n-6)
    x = np.arange(7)
    for i in range(n-6):
        gr[i] = np.polyfit(x, y[i:i+7], 1)[0]
    return gr


# plot cases and deaths over time
def plot_cd_county(ax, cd, county, state, date, pop):
    # number of days to plot and extrapolate
    n_days = cd.shape[0]
    n_days_ext = 10
    n_days_plot = 40

    # normalize by population (cases/deaths per 100,000)
    c0 = cd[:, 0] / pop * 1e5
    d0 = cd[:, 1] / pop * 1e5

    # fit sigmoid to cases and deaths
    t0 = np.flip(np.arange(n_days) + 1.)
    t1 = np.flip(np.arange(-n_days_ext, n_days) + 1.)
    c1, c1_max = fit_sigmoid(smooth_iir_1(c0), n_days_ext)
    d1, d1_max = fit_sigmoid(smooth_iir_1(d0), n_days_ext)

    # title
    ax.set_title('{} County, {} - {}'.format(county, state, date), {'fontsize': 10})

    # plot cases asymptote
    ax.plot([t1[0], t1[-1]], [c1_max, c1_max], lw=0.5, color='tab:gray')

    # plot deaths asymptote
    ax2 = ax.twinx()
    ax2.plot([t1[0], t1[-1]], [d1_max, d1_max], lw=0.5, color='xkcd:baby pink')

    # plot cases
    ax.plot(t1, c1, color='tab:gray')
    ax.plot(t0, c0, color='k')
    ax.set_ylabel('Cases (per 100k)')
    ax.set_xlabel('Days ago')
    ax.set_xlim(n_days_plot, -n_days_ext)
    ax.set_ylim(0, 1.2 * np.max(c0))

    # plot deaths
    ax2.plot(t1, d1, color='xkcd:baby pink')
    ax2.plot(t0, d0, color='tab:red')
    ax2.set_ylabel('Deaths (per 100k)', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_ylim(0, max(2 * np.max(d0), 20))


# plot new cases versus total cases over time
def plot_nc_county(ax, cd, county, state, date, pop):
    # number of days to plot and extrapolate
    n_days = cd.shape[0]
    n_days_ext = 10
    c0 = cd[:, 0]
    c0s = smooth_iir_1(c0)

    # fit sigmoid to cases
    t0 = np.flip(np.arange(n_days) + 1.)
    t1 = np.flip(np.arange(-n_days_ext, n_days) + 1.)
    c1, c1_max = fit_sigmoid(c0s, n_days_ext)

    # get new cases per day
    dc0 = np.diff(c0s)
    dc1 = np.diff(c1)
    c0 = c0[1:]
    c1 = c1[1:]

    # find cutoff
    idx = max(np.argmax(dc0 > 1), np.argmax(c0 > 1))
    dc0 = dc0[idx:]
    dc1 = dc1[idx:]
    c0 = c0[idx:]
    c1 = c1[idx:]

    # plot cases
    ax.loglog(c1, dc1, color='tab:gray')
    ax.loglog(c0, dc0, color='k', marker='.')
    ax.loglog(c0[-1], dc0[-1], color='r', marker='o')
    ax.set_ylabel('Cases Per Day (smoothed)')
    ax.set_xlabel('Total Cases')


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
        plot_cd_county(axs[0, i], cd, county, state, date, pop)
        plot_nc_county(axs[1, i], cd, county, state, date, pop)


# table of new deaths per day for the top 10 states
def plot_nd_states():
    states, cd_states, date_str = load_cd_states()
    n_states = len(states)
    # compute growth rates and their corresponding scores
    gr = []
    gr_score = np.zeros(n_states)
    for i in range(n_states):
        pop = load_pop_state(states[i])
        gr.append(est_growth_rate(smooth_iir_1(cd_states[i][:,1])) / pop * 1e5)
        gr_score[i] = np.mean(gr[i][-10:]) + np.max(gr[i])  + 10 * (pop > 2e5)
    # sort based on recent growth rate
    idx = np.argsort(gr_score)
    gr = [gr[i] for i in idx]
    # plot
    n_top = 10
    fig = plt.figure(figsize=(8, 5))
    x = np.arange(60, 0, -1)
    # uncomment to plot all states
    # for i in range(n_states-n_top):
    #     n = min(len(gr[i]), 60)
    #     plt.plot(x[-n:], gr[i][-n:], color='tab:gray', lw=0.5)
    for i in range(n_top-1,-1,-1):
        n = min(len(gr[n_states-n_top+i]), 60)
        plt.plot(x[-n:], gr[n_states-n_top+i][-n:], label=states[idx[n_states-n_top+i]], lw=2.0)
    plt.xlabel('Days Ago')
    plt.ylabel('New Deaths per 100k (Weekly Average)')
    plt.xlim(60, 0)
    plt.legend(loc='upper left', fontsize='small')
    plt.title(date_str)


if __name__ == "__main__":
    plot_county_list([('Santa Clara', 'California'), ('Los Angeles', 'California'),  ('Broward', 'Florida')])
    #plot_county_list([('Pima', 'Arizona'), ('Mecklenburg', 'North Carolina'), ('Travis', 'Texas')])
    plot_nd_states()
    plt.show()

