
import casadi as cd
import numpy as np
# from circle_fit import taubinSVD
from scipy.integrate import odeint,solve_ivp,ode
# import cvxpy as cp



lf = 1.0
lr = 1.0
a0 = 0
a1 = 0
a2 = 0
Mass = 1
dt = 0.15
beta = 0

def bezier_curve(t, P0, P1, P2, P3):
    return (1-t)**3 * P0 + 3*(1-t)**2 * t * P1 + 3*(1-t) * t**2 * P2 + t**3 * P3
def dynamics(t, x):
    dx = [0] * 6
    beta = 0
    dx[0] = x[3] * np.cos(x[2] + beta)
    dx[1] = x[3] * np.sin(x[2] + beta)
    dx[2] = (x[3] * x[5]) / (lf + lr)
    dx[3] = -1/Mass * (a0*np.sign(x[3])+ a1*x[3] + a2*x[3]**2) + 1 / Mass * x[4]
    dx[4] = 0
    dx[5] = 0
    return dx


def dynamics1(t, x):
    dx = [0] * 6
    beta = 0
    dx[0] = x[3] * np.cos(x[2])
    dx[1] = x[3] * np.sin(x[2])
    dx[2] = (x[3] * x[5]) / (lf + lr)
    dx[3] = x[4]
    dx[4] = 0
    dx[5] = 0
    return dx

def motion(dynamics, state, Input, timespan, teval):
    y0 = np.append(state, Input)
    sol = solve_ivp(dynamics, timespan, y0, method = 'DOP853', t_eval=[teval], atol=1e-6)
    x = np.reshape(sol.y[0:len(state)], len(state))
    return x

def rk4( t, state, Input, n):
    state = np.append(state, Input)
    # Calculating step size
    # x0 = np.append(state, Input)
    h = np.array([(t[-1] - t[0]) / n])
    t0 = t[0]
    for i in range(n):
        k1 = np.array(dynamics(t0, state))
        k2 = np.array(dynamics((t0 + h / 2), (state + h * k1 / 2)))
        k3 = np.array(dynamics((t0 + h / 2), (state + h * k2 / 2)))
        k4 = np.array(dynamics((t0 + h), (state + h * k3)))
        k = np.array(h * (k1 + 2 * k2 + 2 * k3 + k4) / 6)
        # k = np.array(h * (k1))
        xn = state + k
        state = xn
        t0 = t0 + h

    return xn[0:4]

def mpc_exec(Ref_Rightlane,Ref_Leftlane,Ref_Centerlane,Psi_ref,last_psi_ref_path, states, states_ip):
    statesnumber = 4
    Inputnumber = 2
    N = 5
    params = 20 * [0.5]
    x0,y0,psi0,v0 = states
    opti = cd.Opti()
    umin = -0.6 * 9.81 * Mass
    umax = 0.4 * 9.81 * Mass

    steermin = -0.6
    steermax = 0.6

    X = opti.variable(statesnumber, N + 1)
    u = opti.variable(Inputnumber + 2, N)
    s = opti.variable(4, N)

    # "Fit a circle"
    # xcr, ycr, Rr, sigma = taubinSVD(Ref_Rightlane)
    # xcl, ycl, Rl, sigma = taubinSVD(Ref_Leftlane)


    Ref_Centerlane_x = [Ref_Centerlane[i][0] for i in range(0, len(Ref_Centerlane))]
    Ref_Centerlane_y = [Ref_Centerlane[i][1] for i in range(0, len(Ref_Centerlane))]

    waypoints2centerx = [x0] + Ref_Centerlane_x[2:8]
    waypointstraj2centery = [y0] + Ref_Centerlane_y[2:8]

    x1 = np.linspace(waypoints2centerx[0], waypoints2centerx[-1], len(Psi_ref))
    y1 = np.linspace(waypointstraj2centery[0], waypointstraj2centery[-1], len(Psi_ref))
    Psi_ref_path = [np.arctan2((y1[i + 1] - y1[i]), (x1[i + 1] - x1[i])) for i in range(len(Psi_ref) - 1)]
    Psi_ref_path_continous = [
        (psi_ref - last_psi_ref_path + np.pi) % (2 * np.pi) - np.pi + last_psi_ref_path
        for psi_ref in Psi_ref_path]
    # last_psi_ref_path = Psi_ref_path_continous[-1]
    last_psi_ref_path = Psi_ref[0]

    right_lane_x_prioritize = Ref_Rightlane[:5, 0]
    right_lane_y_prioritize = Ref_Rightlane[:5, 1]

    # Fit a line to the right lane boundaries, prioritizing the first few points
    right_coefficients_prioritize = np.polyfit(right_lane_x_prioritize, right_lane_y_prioritize, 1,
                                               w=np.arange(1, 5 + 1))

    # Fit a line to the left lane boundaries
    left_lane_x_prioritize = Ref_Leftlane[:5, 0]
    left_lane_y_prioritize = Ref_Leftlane[:5, 1]
    left_coefficients_prioritize = np.polyfit(left_lane_x_prioritize, left_lane_y_prioritize, 1,
                                               w=np.arange(1, 5 + 1))

    for k in range(0, N):
        beta = np.arctan(0.5 * np.tan(u[1, k]))
        if right_coefficients_prioritize[0] <= 0.01 and right_coefficients_prioritize[0] >= -0.01:
            if y0 - Ref_Rightlane[0][1] >= 0:
                b1 = X[1, k] - Ref_Rightlane[0][1]
                lfb1 = X[3, k] * cd.sin(X[2, k] + beta) + params[0] * b1
                l2fb1 = (params[0] * (lfb1 - params[0] * b1) + params[1] * lfb1
                         -1/Mass * cd.sin(X[2, k] + beta) *(a0*np.sign(X[3, k] )+ a1*X[3, k]  + a2*X[3, k] **2))
                lgu1 = cd.sin(X[2, k] + beta) *  1 / Mass
                lgdelta1 = X[3, k] * cd.cos(X[2, k] + beta) * (X[3, k]) / (lf + lr)
                opti.subject_to(l2fb1 + lgu1 * u[0, k] + lgdelta1 * u[1, k] + s[0, k] >= 0)

                b2 = -X[1, k] + Ref_Leftlane[0][1]
                lfb2 = -X[3, k] * cd.sin(X[2, k] + beta) + params[2] * b2
                l2fb2 = (params[2] * (lfb2 - params[2] * b2) + params[3] * lfb2 +
                         1/Mass *  cd.sin(X[2, k] + beta) *(a0*np.sign(X[3, k])+ a1*X[3, k]+ a2*X[3, k] **2))
                lgu2 = -cd.sin(X[2, k] + beta)* 1 / Mass
                lgdelta2 = -X[3, k] * cd.cos(X[2, k] + beta) * (X[3, k]) / (lf + lr)
                opti.subject_to(l2fb2 + lgu2 * u[0, k] + lgdelta2 * u[1, k] + s[1, k] >= 0)

            else:
                b1 = X[1, k] - Ref_Leftlane[0][1]
                lfb1 = X[3, k] * cd.sin(X[2, k] + beta) + params[4] * b1
                l2fb1 = (params[4] * (lfb1 - params[4] * b1) + params[5] * lfb1  -
                        1/Mass * cd.sin(X[2, k] + beta)*(a0*np.sign(X[3, k])+ a1*X[3, k]+ a2*X[3, k] **2))
                lgu1 = cd.sin(X[2, k] + beta) * 1 /Mass
                lgdelta1 = X[3, k] * cd.cos(X[2, k] + beta) * (X[3, k]) / (lf + lr)
                opti.subject_to(l2fb1 + lgu1 * u[0, k] + lgdelta1 * u[1, k] + s[0, k] >= 0)

                b2 = -X[1, k] + Ref_Rightlane[0][1]
                lfb2 = -X[3, k] * cd.sin(X[2, k] + beta) + params[6] * b2
                l2fb2 = (params[6] * (lfb2 - params[6] * b2) + params[7] * lfb2 +
                        1/Mass * cd.sin(X[2, k] + beta)* (a0*np.sign(X[3, k])+ a1*X[3, k]+ a2*X[3, k] **2))
                lgu2 = -cd.sin(X[2, k] + beta) * 1 /Mass
                lgdelta2 = -X[3, k] * cd.cos(X[2, k] + beta) * (X[3, k]) / (lf + lr)
                opti.subject_to(l2fb2 + lgu2 * u[0, k] + lgdelta2 * u[1, k] + s[1, k] >= 0)

        elif right_coefficients_prioritize[0] >= 75 or right_coefficients_prioritize[0] <= -75:

            if x0 - Ref_Rightlane[0][0] >= 0:
                b1 = X[0, k] - Ref_Rightlane[0][0]
                lfb1 = X[3, k] * cd.cos(X[2, k] + beta) + params[0] * b1
                l2fb1 = (params[0] * (lfb1 - params[0] * b1) + params[1] * lfb1
                         - 1/Mass * cd.cos(X[2, k] + beta)*(a0*np.sign(X[3, k])+ a1*X[3, k]+ a2*X[3, k] **2))
                lgu1 = cd.cos(X[2, k] + beta) * 1/Mass
                lgdelta1 = -X[3, k] * cd.sin(X[2, k] + beta) * (X[3, k]) / (lf + lr)
                opti.subject_to(l2fb1 + lgu1 * u[0, k] + lgdelta1 * u[1, k] + s[0, k] >= 0)

                b2 = -X[0, k] + Ref_Leftlane[0][0]
                lfb2 = -X[3, k] * cd.cos(X[2, k] + beta) + params[2] * b2
                l2fb2 = (params[2] * (lfb2 - params[2] * b2) + params[3] * lfb2 +
                         1/Mass * cd.cos(X[2, k] + beta) *(a0*np.sign(X[3, k])+ a1*X[3, k]+ a2*X[3, k] **2))
                lgu2 = -cd.cos(X[2, k] + beta)* 1/Mass
                lgdelta2 = X[3, k] * cd.sin(X[2, k] + beta) * (X[3, k]) / (lf + lr)
                opti.subject_to(l2fb2 + lgu2 * u[0, k] + lgdelta2 * u[1, k] + s[1, k] >= 0)

            else:
                b1 = X[0, k] - Ref_Leftlane[0][0]
                lfb1 = X[3, k] * cd.cos(X[2, k] + beta) + params[4] * b1
                l2fb1 = (params[4] * (lfb1 - params[4] * b1) + params[5] * lfb1 -
                         1/Mass * cd.cos(X[2, k] + beta)*(a0*np.sign(X[3, k])+ a1*X[3, k]+ a2*X[3, k] **2))
                lgu1 = cd.cos(X[2, k] + beta)* 1/Mass
                lgdelta1 = -X[3, k] * cd.sin(X[2, k] + beta) * (X[3, k]) / (lf + lr)
                opti.subject_to(l2fb1 + lgu1 * u[0, k] + lgdelta1 * u[1, k] + s[0, k] >= 0)

                b2 = -X[0, k] + Ref_Rightlane[0][0]
                lfb2 = -X[3, k] * cd.cos(X[2, k] + beta) + params[6] * b2
                l2fb2 = (params[6] * (lfb2 - params[6] * b2) + params[7] * lfb2
                         + 1/Mass * cd.cos(X[2, k] + beta)*(a0*np.sign(X[3, k])+ a1*X[3, k]+ a2*X[3, k] **2))
                lgu2 = -cd.cos(X[2, k] + beta)* 1/Mass
                lgdelta2 = X[3, k] * cd.sin(X[2, k] + beta) * (X[3, k]) / (lf + lr)
                opti.subject_to(l2fb2 + lgu2 * u[0, k] + lgdelta2 * u[1, k] + s[1, k] >= 0)
        else:
            if y0 - right_coefficients_prioritize[0] * x0 - right_coefficients_prioritize[1] >= 0:
                b1 = X[1, k] - right_coefficients_prioritize[0] * X[0, k] - right_coefficients_prioritize[1]
                lfb1 = (X[3, k] * cd.sin(X[2, k] + beta) - right_coefficients_prioritize[0] * (X[3, k] * cd.cos(X[2, k] + beta))
                        + params[0] * b1)
                l2fb1 = ((params[0] * (lfb1 - params[0] * b1) + params[1] * lfb1 -
                         1/Mass * cd.sin(X[2, k] + beta)*(a0*np.sign(X[3, k])+ a1*X[3, k]+ a2*X[3, k] **2))
                         + right_coefficients_prioritize[0] * 1/Mass * cd.cos(X[2, k] + beta) * (a0*np.sign(X[3, k])+ a1*X[3, k]+ a2*X[3, k] **2))
                lgu1 = 1/Mass*(cd.sin(X[2, k] + beta) - right_coefficients_prioritize[0] * cd.cos(X[2, k] + beta))
                lgdelta1 = (X[3, k] * cd.cos(X[2, k] + beta) * (X[3, k]) / (lf + lr) + right_coefficients_prioritize[0] *
                            (X[3, k] * cd.sin(X[2, k] + beta) * (X[3, k]) / (lf + lr)))
                opti.subject_to(l2fb1 + lgu1 * u[0, k] + lgdelta1 * u[1, k] + s[0, k] >= 0)

                b2 = -X[1, k] + left_coefficients_prioritize[0] * X[0, k] + left_coefficients_prioritize[1]
                lfb2 = (-X[3, k] * cd.sin(X[2, k] + beta) + left_coefficients_prioritize[0] * (X[3, k] * cd.cos(X[2, k] + beta))
                        + params[0] * b2)
                l2fb2 = params[2] * (lfb2 - params[2] * b2) + params[3] * lfb2 + 1/Mass * cd.sin(X[2, k] + beta)*(a0*np.sign(X[3, k])+ a1*X[3, k]+ a2*X[3, k] **2)
                - left_coefficients_prioritize[0] * cd.cos(X[2, k] + beta) * 1/Mass *(a0*np.sign(X[3, k])+ a1*X[3, k]+ a2*X[3, k] **2)
                lgu2 = 1/Mass*(-cd.sin(X[2, k] + beta) + left_coefficients_prioritize[0] * cd.cos(X[2, k] + beta))
                lgdelta2 = (-X[3, k] * cd.cos(X[2, k] + beta) * (X[3, k]) / (lf + lr) - left_coefficients_prioritize[0] *
                            (X[3, k] * cd.sin(X[2, k] + beta) * (X[3, k]) / (lf + lr)))

                opti.subject_to(l2fb2 + lgu2 * u[0, k] + lgdelta2 * u[1, k] + s[1, k] >= 0)


            else:
                b1 = X[1, k] - left_coefficients_prioritize[0] * X[0, k] - left_coefficients_prioritize[1]
                lfb1 = (X[3, k] * cd.sin(X[2, k]) - left_coefficients_prioritize[0] * (X[3, k] * cd.cos(X[2, k]))
                        + params[0] * b1)
                l2fb1 = params[0] * (lfb1 - params[0] * b1) + params[1] * lfb1
                lgu1 = cd.sin(X[2, k]) - left_coefficients_prioritize[0] * cd.cos(X[2, k])
                lgdelta1 = (X[3, k] * cd.cos(X[2, k]) * (X[3, k]) / (lf + lr) + left_coefficients_prioritize[0] *
                            (X[3, k] * cd.sin(X[2, k]) * (X[3, k]) / (lf + lr)))
                opti.subject_to(l2fb1 + lgu1 * u[0, k] + lgdelta1 * u[1, k] + s[0, k] >= 0)

                b2 = -X[1, k] + right_coefficients_prioritize[0] * X[0, k] + right_coefficients_prioritize[1]
                lfb2 = (-X[3, k] * cd.sin(X[2, k]) + right_coefficients_prioritize[0] * (X[3, k] * cd.cos(X[2, k]))
                        + params[0] * b2)
                l2fb2 = params[2] * (lfb2 - params[2] * b2) + params[3] * lfb2
                lgu2 = -cd.sin(X[2, k]) + right_coefficients_prioritize[0] * cd.cos(X[2, k])
                lgdelta2 = (-X[3, k] * cd.cos(X[2, k]) * (X[3, k]) / (lf + lr) - right_coefficients_prioritize[0] *
                            (X[3, k] * cd.sin(X[2, k]) * (X[3, k]) / (lf + lr)))
                opti.subject_to(l2fb2 + lgu2 * u[0, k] + lgdelta2 * u[1, k] + s[1, k] >= 0)


    curr_states = states_ip
    for k in range(0,N):
        a = 3
        b = 2
        xip, yip, psi_ip, vip = curr_states
        b3 = (X[0, k] - xip) ** 2 / a + (X[1, k] - yip) ** 2 / b - X[3, k] ** 2 - 4**2
        lfb3 = (2 * ((X[0, k] - xip) ** 2 / a + (X[1, k] - yip) ** 2 / b - X[3, k] ** 2 - 4**2) +
                (X[3, k] * cd.cos(X[2, k]) * (2 * X[0, k] - 2 * xip)) / a -
                (vip * cd.cos(psi_ip) * (2 * X[0, k] - 2 * xip)) / a +
                (X[3, k] * cd.sin(X[2, k]) * (2 * X[1, k] - 2 * yip)) / b -
                (vip * cd.sin(psi_ip) * (2 * X[1, k] - 2 * yip)) / b) + 2 * (X[3, k]) * (+1/Mass * (a0*cd.sign(X[3, k])+ a1*X[3, k] + a2*X[3, k]**2))
        lgb3u = -2 * X[3, k] * 1 / Mass
        lgb3delta = 0

        opti.subject_to(lfb3 + lgb3u * u[0, k] + lgb3delta * u[1, k] + s[2, k] >= 0)
        curr_states = motion(dynamics, curr_states, [0, 0], [0, dt], dt)
        if k == 0:
            b31 = b3
            lfb31 = lfb3
            lgb3u1 = lgb3u
            lgb3delta1 = lgb3delta


    for k in range(0, N):
        b = X[3, k] - 0
        lfb = params[14] * b
        lfb = 1 * b
        lgu = 1
        opti.subject_to(lfb + lgu * u[0, k] >= 0)

        b = -X[3, k] + 30
        lfb = params[15] * b
        lfb = 1 * b
        lgu = -1
        opti.subject_to(lfb + lgu * u[0, k] >= 0)

    # Define the cost function
    cost = 0
    diag_elements_u = [params[16], params[17], params[18], params[19]]
    u_ref = np.zeros((Inputnumber + 2, N))


    normalization_factor = [max(-umin, umax), max(steermax, -steermin), 100,
                            0.4]
    for i in range(Inputnumber + 2):
        for h in range(N):
            cost += 0.5 * diag_elements_u[i] * ((u[i, h] - u_ref[i][h]) / normalization_factor[i]) ** 2
            cost += 10 ** 8 * (s[0, h]) ** 2
            cost += 10 ** 8 * (s[1, h]) ** 2
            cost += 10 ** 8 * (s[2, h]) ** 2
            cost += 10 ** 8 * (s[3, h]) ** 2

    opti.subject_to(umin <= u[0, :])
    opti.subject_to(u[0, :] <= umax)
    eps3 = 1
    for k in range(0, N):
        v_des = 15
        V = (X[3, k] - v_des) ** 2
        #without friction
        # lfV = eps3 * V
        # lgu = 2 * (X[3, k] - 15)
        # with friction
        lfV = eps3 * V + 2 * (X[3, k] - v_des) * (-1/Mass * (a0*cd.sign(X[3, k])+ a1*X[3, k] + a2*X[3, k]**2))
        lgu = 2 * (X[3, k] - v_des) * 1/Mass
        opti.subject_to(lfV + lgu * u[0, k] - u[2, k] <= 0)

        V = (X[2, k] - Psi_ref_path_continous) ** 2
        lfV = 10 * eps3 * V
        lgdelta = 2 * (X[2, k] - Psi_ref_path_continous) * X[3, k] / (lr + lf)
        opti.subject_to(lfV + lgdelta * u[1, k] - u[3, k] <= 0)

    opti.subject_to(steermin <= u[1, :])
    opti.subject_to(u[1, :] <= steermax)

    opti.subject_to(X[:, 0] == states)  # initialize states
    timespan = [0, dt]

    for h in range(N):  # initial guess
        opti.set_initial(X[:, h], [Ref_Centerlane[h][0], Ref_Centerlane[h][1], Psi_ref[h], v0])

    for k in range(N):
        state = []
        Input = []
        for j in range(statesnumber):
            state.append(X[j, k])
        for j in range(0, Inputnumber):
            Input.append(u[j, k])
        state = rk4(timespan, state, Input, 1)
        for j in range(statesnumber):
            opti.subject_to(X[j, k + 1] == state[j])


    opts = {}
    opts['print_time'] = False
    opts['ipopt.print_level'] = False
    opti.solver('ipopt', opts)
    opti.minimize(cost)
    sol = opti.solve()

    if N > 1:
        # if road == 0:
        #     print(sol.value(b41),params[12])
        # else:
        #     print(sol.value(b41), params[13])
        # RightCBF = sol.value(l2fb11 + lgu11 * sol.value(u)[0, 0] + lgdelta11 * sol.value(u)[1, 0])
        # LeftCBF = sol.value(l2fb21 + lgu21 * sol.value(u)[0, 0] + lgdelta21 * sol.value(u)[1, 0])
        RearendCBF = sol.value(lfb31 + lgb3u1 * sol.value(u)[0, 0] + lgb3delta1 * sol.value(u)[1, 0])
        # print( sol.value((X[0, 0] - xip) ** 2 / a), sol.value((X[1, 0] - yip) ** 2 / b))
        #print(sol.value(b31), RearendCBF)
        # print("MPC sol:", sol.value(X[3,1]))
        # MergingCBF = sol.value(lfb41 + lgb4u1 * sol.value(u)[0, 0] + lgb4delta1 * sol.value(u)[1, 0])
        #
        # right_cbf = b1
        # # # left_cbf = b2
        # RightCBF.append(RightCBF)
        # LeftCBF.append(LeftCBF)
        # # RearendCBF.append(RearendCBF)
        # MergingCBF.append(MergingCBF)
        # LeftOrg.append(sol.value(b11))
        # RightOrg.append(sol.value(b21))
        # MergingOrg.append(sol.value(b41))
        # # RearendOrg.append(sol.value(b31))
        # accdata.append(sol.value(u)[0, 0])
        # steerdata.append(sol.value(u)[1, 0])
        # obj.append(sol.value(cost))
        s_vars = sol.value(s)[:, 0]
        # if np.any(s_vars > 0.1):  # or LeftCBF <= 0 or RightCBF <= 0:
        #     s_vars = [min(sol.value(s)[i, 0], 0.1) for i in range(len(sol.value(s)[:, 0]))]
        #     # s_vars = sol.value(s)[2, 0]
        #     return "No solution found", np.array([-5.66, 0])
        # else:
        # print(np.arctan(0.5*np.tan(sol.value(u)[1, 0])))
        acc = 1 / Mass * sol.value(u)[0, 0]
        model_acc = -1 / Mass * (a0 * np.sign(v0) + a1 * v0 + a2* v0 ** 2) + 1 / Mass * sol.value(u)[0, 0]
        return "solution found", [acc, sol.value(u)[1, 0]], sol.value(X)[:,1], model_acc
    else:

        # RightCBF = sol.value(l2fb11 + lgu11 * sol.value(u[0]) + lgdelta11 * sol.value(u[1]))
        # LeftCBF = sol.value(l2fb21 + lgu21 * sol.value(u[0]) + lgdelta21 * sol.value(u[1]))
        # RearendCBF = sol.value(lfb31 + lgb3u1 * sol.value(u[0]) + lgb3delta1 * sol.value(u[1]))
        # MergingCBF = sol.value(lfb41 + lgb4u1 * sol.value(u[0]) + lgb4delta1 * sol.value(u[1]))
        #
        # RightCBF.append(RightCBF)
        # LeftCBF.append(LeftCBF)
        # RearendCBF.append(RearendCBF)
        # MergingCBF.append(MergingCBF)
        # LeftOrg.append(sol.value(b11))
        # RightOrg.append(sol.value(b21))
        # MergingOrg.append(sol.value(b31))
        # RearendOrg.append(sol.value(b41))
        # vehicle.RightCBF.append(sol.value(b11))
        # vehicle.LeftCBF.append(sol.value(b21))
        s_vars = sol.value(s)
        if np.any(sol.value(s) > 0.1):  # or LeftCBF <= 0 or RightCBF <= 0:
            s_vars = [0.1] * 4
            acc = -6
            model_acc = -1 / 1650 * (0.1 * np.sign(v0) + 5 * v0 + 0.25 * v0 ** 2) + 1 / 1650 * acc
            next_states = motion(dynamics, states, [acc, 0], [0, dt], dt)
            return "No solution found", np.array([acc, 0]), next_states, model_acc
        else:
            return "solution found", sol.value(u[:2]),sol.value(X)[:,1], -1


def mpc_exec_highway(agent_dict, acc_ref, states, preceding_cars_state, conflicting_cars_state_deg1, conflicting_cars_state_deg2, vid, vid_select, current_lane_index, non_cavs_lane_ids, MPs,conflicting_cars_state_deg3, scenario='highway'):

    Psi_ref = agent_dict["ref_theta"]
    centerlane = agent_dict["ref_centerlane"]
    closest_index = round(agent_dict["completion"] * agent_dict["length_path"])
    closest_index = min(closest_index, len(Psi_ref) - 8)
    Psi_ref = Psi_ref[closest_index:]

    veh_len = 1
    dt = 0.1
    eps = 1
    statesnumber = 4
    Inputnumber = 2
    N = 1

    params = 20 * [0.5]
    x0, y0, psi0, v0 = states
    opti = cd.Opti()
    umin = -0.6 * 9.81 * Mass
    umax = 0.4 * 9.81 * Mass

    steermin = -0.6 * (35 - v0)/35
    steermax = 0.6 * (35 - v0)/35

    X = opti.variable(statesnumber, N + 1)
    u = opti.variable(Inputnumber + 2, N)
    s = opti.variable(4, N)

    left_most_boundary = 135.5
    right_most_lane = 150.75


    maximum_noise_x = 5
    maximum_noise_v = 5
    b3_ic = []

    if current_lane_index != agent_dict["reference_lane_index"]:
        for vehicle_states in conflicting_cars_state_deg1['front']:
            curr_states = vehicle_states
            for k in range(0, N):
                a_x = 1
                a_y = 0.7
                k1 = 10
                x_ic, y_ic, psi_ic, v_ic = curr_states
                y_ic = 0.3 * y_ic + 0.7*y0
                if x_ic < x0:
                    x_ic = min(x_ic + maximum_noise_x, x0)
                    v_ic_2 = v_ic + maximum_noise_v
                else:
                    x_ic = max(x_ic - maximum_noise_x, x0)
                    v_ic_2 = v_ic - maximum_noise_v

                if y_ic < y0:
                    v_ic_1 = v_ic + maximum_noise_v
                else:
                    v_ic_1 = v_ic - maximum_noise_v

                # agent_dict['ellipsoid']['b'] = a_y * (eps + v0)
                # agent_dict['ellipsoid']['a'] = a_x * (eps + v0)
                b3_front = -(veh_len+1)  ** 2 + (X[1, k] - y_ic) ** 2 / (a_y ** 2 * (eps + X[3, k]) ** 2) + (X[0, k] - x_ic) ** 2 / (a_x ** 2 * (eps + X[3, k]) ** 2)

                lfb3_front = (k1 * (-(veh_len+1) ** 2 + (X[1, k] - y_ic) ** 2 / (a_y ** 2 * (eps + X[3, k]) ** 2) + (X[0, k] - x_ic) ** 2 / (a_x ** 2 * (eps + X[3, k]) ** 2)) +
                        2 * (a_x ** 2 * (X[1, k] - y_ic) * ( X[3, k] * cd.sin(beta + X[2, k]) - v_ic_1 * cd.sin(beta + psi_ic)) +
                             a_y ** 2 * (X[0, k] - x_ic) * (X[3, k] * cd.cos(beta + X[2, k]) - v_ic_2 * cd.cos(beta + psi_ic))) / (a_x ** 2 * a_y ** 2 * (eps + X[3, k]) ** 2))
                lgb3delta_front = 0
                lgb3u_front = (-2 * (X[1, k] - y_ic) ** 2 / (a_y ** 2 * (eps + X[3, k]) ** 3) - 2 * (X[0, k] - x_ic) ** 2 / (a_x ** 2 * (eps + X[3, k]) ** 3)) / Mass

                opti.subject_to(lfb3_front + lgb3u_front * u[0, k] + lgb3delta_front * u[1, k] + s[1, k] >= 0)
                curr_states = motion(dynamics, curr_states, [0, 0], [0, dt], dt)
                if k == 0:
                    # if vid == vid_select:
                    #     states_i = states
                    states_icc = [x_ic, y_ic, psi_ic,v_ic]
                    b3_ic.append(b3_front)
                    # lfb3_ic = min(lfb3_ic, lfb3)
                    lgb3u_ic = lgb3u_front
                    # lgb3delta_ic = lgb3delta
                    # cbf_ic = lfb3_ic + lgb3u_ic * u[0, k] + lgb3delta_ic * u[1, k]
               
        for vehicle_states in conflicting_cars_state_deg1['rear']:
            curr_states = vehicle_states
            for k in range(0, N):
                a_x = 1
                a_y = 0.7
                k1 = 10
                x_ic, y_ic, psi_ic, v_ic = curr_states
                y_ic = 0.7 * y_ic + 0.3*y0
                if x_ic < x0:
                    x_ic = min(x_ic + maximum_noise_x, x0)
                    v_ic_2 = v_ic + maximum_noise_v
                else:
                    x_ic = max(x_ic - maximum_noise_x, x0)
                    v_ic_2 = v_ic - maximum_noise_v
                if y_ic < y0:
                    v_ic_1 = v_ic + maximum_noise_v
                else:
                    v_ic_1 = v_ic - maximum_noise_v
                # agent_dict['ellipsoid']['b'] = a_y * (eps + v0)
                # agent_dict['ellipsoid']['a'] = a_x * (eps + v0)
                b3_rear = -(veh_len+1) ** 2 + (-X[1, k] + y_ic) ** 2 / (a_y ** 2 * (eps + X[3, k]) ** 2) + (-X[0, k] + x_ic) ** 2 / (a_x ** 2 * (eps + X[3, k]) ** 2)
                lfb3_rear = (k1 * (-(veh_len+1) ** 2 + (-X[1, k] + y_ic) ** 2 / (a_y ** 2 * (eps + X[3, k]) ** 2) + (-X[0, k] + x_ic) ** 2 / (a_x ** 2 * (eps + X[3, k]) ** 2)) +
                        2 * (a_x ** 2 * (-X[1, k] + y_ic) * ( -X[3, k] * cd.sin(beta + X[2, k]) + v_ic_1 * cd.sin(beta + psi_ic)) +
                             a_y ** 2 * (-X[0, k] + x_ic) * (-X[3, k] * cd.cos(beta + X[2, k]) + v_ic_2 * cd.cos(beta + psi_ic))) / (a_x ** 2 * a_y ** 2 * (eps + X[3, k]) ** 2))
                lgb3delta_rear = 0
                lgb3u_rear = (2 * (X[1, k] - y_ic) ** 2 / (a_y ** 2 * (eps + X[3, k]) ** 3) + 2 * (X[0, k] - x_ic) ** 2 / (a_x ** 2 * (eps + X[3, k]) ** 3)) / Mass

                opti.subject_to(lfb3_rear + lgb3u_rear * u[0, k] + lgb3delta_rear * u[1, k] + s[2, k] >= 0)
                curr_states = motion(dynamics, curr_states, [0, 0], [0, dt], dt)
                if k == 0:
                    # if vid == vid_select:
                    #     states_i = states
                    states_icc = [x_ic, y_ic, psi_ic,v_ic]
                    b3_ic.append(b3_rear)
                    # lfb3_ic = min(lfb3_ic, lfb3)
                    lgb3u_ic = lgb3u_rear
                    # lgb3delta_ic = lgb3delta
                    # cbf_ic = lfb3_ic + lgb3u_ic * u[0, k] + lgb3delta_ic * u[1, k]
        
        # for vehicle_states in conflicting_cars_state_deg2:
        #     curr_states = vehicle_states
        #     for k in range(0, N):
        #         a_x = 1
        #         a_y = 0.15
        #         k1 = 10
        #         x_ic, y_ic, psi_ic, v_ic = curr_states
        #         b3 = -veh_len ** 2 + (X[1, k] - y_ic) ** 2 / (a_y ** 2 * (eps + X[3, k]) ** 2) + (
        #                     X[0, k] - x_ic) ** 2 / (a_x ** 2 * (eps + X[3, k]) ** 2)
        #         lfb3 = k1 * (-veh_len ** 2 + (X[1, k] - y_ic) ** 2 / (a_y ** 2 * (eps + X[3, k]) ** 2) + (
        #                     X[0, k] - x_ic) ** 2 / (a_x ** 2 * (eps + X[3, k]) ** 2)) + 2 * (
        #                               a_x ** 2 * (X[1, k] - y_ic) * (
        #                                   X[3, k] * cd.sin(beta + X[2, k]) - v_ic * cd.sin(beta + psi_ic)) + a_y ** 2 * (
        #                                           X[0, k] - x_ic) * (
        #                                           X[3, k] * cd.cos(beta + X[2, k]) - v_ic * cd.cos(beta + psi_ic))) / (
        #                               a_x ** 2 * a_y ** 2 * (eps + X[3, k]) ** 2)
        #         lgb3delta = 0
        #         lgb3u = (-2 * (X[1, k] - y_ic) ** 2 / (a_y ** 2 * (eps + X[3, k]) ** 3) - 2 * (
        #                     X[0, k] - x_ic) ** 2 / (a_x ** 2 * (eps + X[3, k]) ** 3)) / Mass
        #         opti.subject_to(lfb3 + lgb3u * u[0, k] + lgb3delta * u[1, k] + s[1, k] >= 0)
        #         curr_states = motion(dynamics, curr_states, [0, 0], [0, dt], dt)

    for vehicle_states in preceding_cars_state:
        curr_states = vehicle_states
        for k in range(0, N):
            a_x = 1
            a_y = 0.02
            k1 = 10
            x_ip, y_ip, psi_ip, v_ip = curr_states
            v_ip = v_ip - maximum_noise_v
            x_ip = x_ip - maximum_noise_x
            agent_dict['ellipsoid']['b'] = a_y * (eps + v0)
            agent_dict['ellipsoid']['a'] = a_x * (eps + v0)
            y_ip = states[1]
            b3 = -(veh_len) ** 2 + (X[1, k] - y_ip) ** 2 / (a_y ** 2 * (eps + X[3, k]) ** 2) + (X[0, k] - x_ip) ** 2 / (
                        a_x ** 2 * (eps + X[3, k]) ** 2)
            lfb3 = (k1 * b3
                    + 2 * (a_x ** 2 * (X[1, k] - y_ip) * (
                    X[3, k] * cd.sin(beta + X[2, k]) - v_ip * cd.sin(beta + psi_ip)) + a_y ** 2 * (X[0, k] - x_ip) * (X[3, k] * cd.cos(beta + X[2, k]) - v_ip * cd.cos(beta + psi_ip))) / (a_x ** 2 * a_y ** 2 * (eps + X[3, k]) ** 2))
            lgb3delta = 0
            lgb3u = (-2 * (X[1, k] - y_ip) ** 2 / (a_y ** 2 * (eps + X[3, k]) ** 3) - 2 * (X[0, k] - x_ip) ** 2 / (a_x ** 2 * (eps + X[3, k]) ** 3)) / Mass
            opti.subject_to(lfb3 + lgb3u * u[0, k] + lgb3delta * u[1, k] + s[0, k] >= 0)
            curr_states = motion(dynamics, curr_states, [0, 0], [0, dt], dt)
            if k == 0:
                b3_ip = b3
                lfb3_ip = lfb3
                lgb3u_ip = lgb3u
                lgb3delta_ip = lgb3delta
                cbf_ip = lfb3_ip + lgb3u_ip * u[0, k] + lgb3delta_ip * u[1, k]

    for k in range(0, N):
        b = X[3, k] - 0
        lfb = 1 * b
        lgu = 1
        opti.subject_to(lfb + lgu * u[0, k] >= 0)
        b = -X[3, k] + 20 # to change vmax to something else, modify 20 to the desired value
        lfb = 1 * b
        lgu = -1
        opti.subject_to(lfb + lgu * u[0, k] >= 0)

    # Define the cost function
    cost = 0
    diag_elements_u = [10 * params[16], params[17], 10 * params[18], 10 * params[19]]
    u_ref = np.zeros((Inputnumber + 2, N))
    u_ref[0] = acc_ref * np.ones((1,N))
    normalization_factor = [max(-umin, umax), max(steermax, -steermin), 100, 0.4]
    for i in range(Inputnumber + 2):
        for h in range(N):
            cost += 0.5 * diag_elements_u[i] * ((u[i, h] - u_ref[i][h]) / normalization_factor[i]) ** 2
            cost += 10 ** 8 * (s[0, h]) ** 2
            cost += 10 ** 8 * (s[1, h]) ** 2
            cost += 10 ** 2 * (s[2, h]) ** 2
            cost += 10 ** 8 * (s[3, h]) ** 2

    opti.subject_to(umin <= u[0, :])
    opti.subject_to(u[0, :] <= umax)
    eps3 = 1
    for k in range(0, N):
        V = (X[2, k] - Psi_ref[k]) ** 2
        lfV = 50 * eps3 * V
        lgdelta = 2 * (X[2, k] - Psi_ref[k]) * X[3, k] / (lr + lf)
        opti.subject_to(lfV + lgdelta * u[1, k] - u[3, k] <= 0)

    opti.subject_to(steermin <= u[1, :])
    opti.subject_to(u[1, :] <= steermax)

    opti.subject_to(X[:, 0] == states)  # initialize states
    timespan = [0, dt]

    for h in range(N):  # initial guess
        opti.set_initial(X[:, h], [x0, y0, Psi_ref[0], v0])

    for k in range(N):
        state = []
        Input = []
        for j in range(statesnumber):
            state.append(X[j, k])
        for j in range(0, Inputnumber):
            Input.append(u[j, k])
        state = rk4(timespan, state, Input, 1)
        for j in range(statesnumber):
            opti.subject_to(X[j, k + 1] == state[j])

    opts = {}
    opts['print_time'] = False
    opts['ipopt.print_level'] = False
    opti.solver('ipopt', opts)
    opti.minimize(cost)
    sol = opti.solve()

    if N > 1:
        s_vars = 0
        acc = 1 / Mass * sol.value(u)[0, 0]
        return "solution found", [acc, sol.value(u)[1, 0]], s_vars
    else:
        if np.any(sol.value(s) > 5):  # or LeftCBF <= 0 or RightCBF <= 0:
            acc = -6

            # if sol.value(s[0]) > 5:
            #     print('ip_infeasible')
            # if sol.value(s[1]) > 5:
            #     print('front_lc_infeasible')
            # if sol.value(s[2]) > 5:
            #     print('rear_lc_infeasible')
            s_vars = 5.1
            return "No solution found", np.array([acc, 0.0]), s_vars
        else:
            s_vars = 0
            acc = 1 / Mass * sol.value(u[0])
            return "solution found", sol.value(u[:2]), s_vars


def mpc_exec_crossing(agent_dict, acc_ref, states, preceding_cars_state, conflicting_cars_state_deg1, conflicting_cars_state_deg2, vid, vid_select, current_lane_index,non_cavs_lane_ids, MPs,conflicting_cars_state_deg3, scenario='highway'):
    
    Psi_ref = agent_dict["ref_theta"]
    centerlane = agent_dict["ref_centerlane"]
    closest_index = round(agent_dict["completion"] * agent_dict["length_path"])
    closest_index = min(closest_index, len(Psi_ref) - 8)
    Psi_ref = Psi_ref[closest_index:]

    veh_len = 1
    dt = 0.1
    eps = 1
    statesnumber = 4
    Inputnumber = 2
    N = 1

    params = 20 * [0.5]
    x0, y0, psi0, v0 = states
    opti = cd.Opti()
    umin = -0.6 * 9.81 * Mass
    umax = 0.4 * 9.81 * Mass

    steermin = -0.6 * (35 - v0)/35
    steermax = 0.6 * (35 - v0)/35

    X = opti.variable(statesnumber, N + 1)
    u = opti.variable(Inputnumber + 2, N)
    s = opti.variable(4, N)

    #Comment_to_Sabbir: you probably need to change this boundary and others to get it working in crossing
    if scenario == 'highway':
        left_most_boundary = 135.5
        right_most_lane = 150.75
    elif scenario == 'crossing':
        # To Sabbir: you most likely need to modify this;
        left_most_boundary = 42.5
        right_most_lane = 50.2
    else:
        raise NotImplementedError("Lane boundary not defined in this scenario")

    maximum_noise_x = 5
    maximum_noise_v = 5
    b3_crossing  =[]
    lfb3_crossing = []
    lgb3u_crossing = []

    if current_lane_index != agent_dict["reference_lane_index"]:
        for vehicle_states in conflicting_cars_state_deg1['front']:
            curr_states = vehicle_states
            for k in range(0, N):
                a_x = 1
                a_y = 0.8
                k1 = 10
                x_ic, y_ic, psi_ic, v_ic = curr_states
                y_ic = 0.5 * y_ic + 0.5*y0 
                if x_ic < x0:
                    x_ic = min(x_ic + maximum_noise_x, x0)
                    v_ic_2 = v_ic + maximum_noise_v
                else:
                    x_ic = max(x_ic - maximum_noise_x, x0)
                    v_ic_2 = v_ic - maximum_noise_v

                if y_ic < y0:
                    v_ic_1 = v_ic + maximum_noise_v
                else:
                    v_ic_1 = v_ic - maximum_noise_v

                b3 = -(veh_len+1) ** 2 + (X[1, k] - y_ic) ** 2 / (a_y ** 2 * (eps + X[3, k]) ** 2) + (X[0, k] - x_ic) ** 2 / (a_x ** 2 * (eps + X[3, k]) ** 2)

                lfb3 = (k1 * (-(veh_len+1) ** 2 + (X[1, k] - y_ic) ** 2 / (a_y ** 2 * (eps + X[3, k]) ** 2) + (X[0, k] - x_ic) ** 2 / (a_x ** 2 * (eps + X[3, k]) ** 2)) +
                        2 * (a_x ** 2 * (X[1, k] - y_ic) * ( X[3, k] * cd.sin(beta + X[2, k]) - v_ic_1 * cd.sin(beta + psi_ic)) +
                                a_y ** 2 * (X[0, k] - x_ic) * (X[3, k] * cd.cos(beta + X[2, k]) - v_ic_2 * cd.cos(beta + psi_ic))) / (a_x ** 2 * a_y ** 2 * (eps + X[3, k]) ** 2))
                lgb3delta = 0
                lgb3u = (-2 * (X[1, k] - y_ic) ** 2 / (a_y ** 2 * (eps + X[3, k]) ** 3) - 2 * (X[0, k] - x_ic) ** 2 / (a_x ** 2 * (eps + X[3, k]) ** 3)) / Mass

                opti.subject_to(lfb3 + lgb3u * u[0, k] + lgb3delta * u[1, k] + s[2, k] >= 0)
                curr_states = motion(dynamics, curr_states, [0, 0], [0, dt], dt)
                
        for vehicle_states in conflicting_cars_state_deg1['rear']:
            curr_states = vehicle_states
            for k in range(0, N):
                a_x = 1
                a_y = 0.8
                k1 = 10
                x_ic, y_ic, psi_ic, v_ic = curr_states
                y_ic = 0.5 * y_ic + 0.5*y0
                if x_ic < x0:
                    x_ic = min(x_ic + maximum_noise_x, x0)
                    v_ic_2 = v_ic + maximum_noise_v
                else:
                    x_ic = max(x_ic - maximum_noise_x, x0)
                    v_ic_2 = v_ic - maximum_noise_v

                if y_ic < y0:
                    v_ic_1 = v_ic + maximum_noise_v
                else:
                    v_ic_1 = v_ic - maximum_noise_v

                b3 = -(veh_len+1) ** 2 + (-X[1, k] + y_ic) ** 2 / (a_y ** 2 * (eps + X[3, k]) ** 2) + (-X[0, k] + x_ic) ** 2 / (a_x ** 2 * (eps + X[3, k]) ** 2)

                lfb3 = (k1 * (-(veh_len+1) ** 2 + (-X[1, k] + y_ic) ** 2 / (a_y ** 2 * (eps + X[3, k]) ** 2) + (-X[0, k] + x_ic) ** 2 / (a_x ** 2 * (eps + X[3, k]) ** 2)) +
                        2 * (a_x ** 2 * (-X[1, k] + y_ic) * ( -X[3, k] * cd.sin(beta + X[2, k]) + v_ic_1 * cd.sin(beta + psi_ic)) +
                                a_y ** 2 * (-X[0, k] + x_ic) * (-X[3, k] * cd.cos(beta + X[2, k]) + v_ic_2 * cd.cos(beta + psi_ic))) / (a_x ** 2 * a_y ** 2 * (eps + X[3, k]) ** 2))
                lgb3delta = 0
                lgb3u = (2 * (X[1, k] - y_ic) ** 2 / (a_y ** 2 * (eps + X[3, k]) ** 3) + 2 * (X[0, k] - x_ic) ** 2 / (a_x ** 2 * (eps + X[3, k]) ** 3)) / Mass

                opti.subject_to(lfb3 + lgb3u * u[0, k] + lgb3delta * u[1, k] + s[2, k] >= 0)
                curr_states = motion(dynamics, curr_states, [0, 0], [0, dt], dt)

    for vehicle_states in preceding_cars_state:
        curr_states = vehicle_states
        for k in range(0, N):
            a_x = 1
            a_y = 0.02
            k1 = 10
            x_ip, y_ip, psi_ip, v_ip = curr_states
            v_ip = v_ip - maximum_noise_v
            x_ip = x_ip - maximum_noise_x
            y_ip = states[1]
            b3 = -veh_len ** 2 + (X[1, k] - y_ip) ** 2 / (a_y ** 2 * (eps + X[3, k]) ** 2) + (X[0, k] - x_ip) ** 2 / (
                        a_x ** 2 * (eps + X[3, k]) ** 2)
            lfb3 = (k1 * b3
                    + 2 * (a_x ** 2 * (X[1, k] - y_ip) * (
                    X[3, k] * cd.sin(beta + X[2, k]) - v_ip * cd.sin(beta + psi_ip)) + a_y ** 2 * (X[0, k] - x_ip) * (X[3, k] * cd.cos(beta + X[2, k]) - v_ip * cd.cos(beta + psi_ip))) / (a_x ** 2 * a_y ** 2 * (eps + X[3, k]) ** 2))
            lgb3delta = 0
            lgb3u = (-2 * (X[1, k] - y_ip) ** 2 / (a_y ** 2 * (eps + X[3, k]) ** 3) - 2 * (X[0, k] - x_ip) ** 2 / (a_x ** 2 * (eps + X[3, k]) ** 3)) / Mass
            opti.subject_to(lfb3 + lgb3u * u[0, k] + lgb3delta * u[1, k] + s[0, k] >= 0)
            curr_states = motion(dynamics, curr_states, [0, 0], [0, dt], dt)
            if k == 0:
                b3_ip = b3
                lfb3_ip = lfb3
                lgb3u_ip = lgb3u
                lgb3delta_ip = lgb3delta
                cbf_ip = lfb3_ip + lgb3u_ip * u[0, k] + lgb3delta_ip * u[1, k]
    
    for index, vehicle_states in enumerate(conflicting_cars_state_deg3['front']):
        curr_states = vehicle_states
        x_ic, y_ic, psi_ic, v_ic = curr_states
        mp = MPs[index]
        d1 = mp[0] - x0
        if x_ic > 0:
            y_ic = y_ic - maximum_noise_x
            v_ic = v_ic - maximum_noise_v
            d2 = y_ic - mp[1]
            # if vid == vid_select:
            #     print(x_ic,y_ic)
        else:
            y_ic = y_ic + maximum_noise_x
            v_ic = v_ic - maximum_noise_v
            d2 = mp[1] - y_ic
            # if vid == vid_select:
            #     print(x_ic,y_ic)

        if d1 > 0:
            for k in range(0, N):
                a_x = 0.7
                a_y = 0.4
                if 0 < d1 < v0**2/(2*5.88) + 4.5:
                    if 0 < d2 < d1 + 7.5:
                        k1 = 0.1
                    else:
                        k1 = 3
                else:
                    k1 = 3
                            
                x_ic, y_ic, psi_ic, v_ic = curr_states
                if x_ic < x0:
                    x_ic = min(x_ic + maximum_noise_x, x0)
                    v_ic_2 = v_ic + maximum_noise_v
                else:
                    x_ic = max(x_ic - maximum_noise_x, x0)
                    v_ic_2 = v_ic - maximum_noise_v
                if y_ic < y0:
                    v_ic_1 = v_ic + maximum_noise_v
                else:
                    v_ic_1 = v_ic - maximum_noise_v
            
                agent_dict['ellipsoid']['b'] = a_y * (eps + v0)
                agent_dict['ellipsoid']['a'] = a_x * (eps + v0)
                b3 = -veh_len ** 2 + (X[1, k] - y_ic) ** 2 / (a_y ** 2 * (eps + X[3, k]) ** 2) + (X[0, k] - x_ic) ** 2 / (a_x ** 2 * (eps + X[3, k]) ** 2)
                
                lfb3 = (k1 * (-veh_len ** 2 + (X[1, k] - y_ic) ** 2 / (a_y ** 2 * (eps + X[3, k]) ** 2) + (X[0, k] - x_ic) ** 2 / (a_x ** 2 * (eps + X[3, k]) ** 2)) +
                        2 * (a_x ** 2 * (X[1, k] - y_ic) * ( X[3, k] * cd.sin(beta + X[2, k]) - v_ic_1 * cd.sin(beta + psi_ic)) +
                                a_y ** 2 * (X[0, k] - x_ic) * (X[3, k] * cd.cos(beta + X[2, k]) - v_ic_2 * cd.cos(beta + psi_ic))) / (a_x ** 2 * a_y ** 2 * (eps + X[3, k]) ** 2))
                lgb3delta = 0
                lgb3u = (-2 * (X[1, k] - y_ic) ** 2 / (a_y ** 2 * (eps + X[3, k]) ** 3) - 2 * (X[0, k] - x_ic) ** 2 / (a_x ** 2 * (eps + X[3, k]) ** 3)) / Mass

                opti.subject_to(lfb3 + lgb3u * u[0, k] + lgb3delta * u[1, k] + s[2, k] >= 0)
                curr_states = motion(dynamics, curr_states, [0, 0], [0, dt], dt)
                if k == 0:
                    b3_crossing = b3
                    lfb3_crossing = lfb3
                    lgb3u_crossing = lgb3u

    for k in range(0, N):
        b = X[3, k] - 0
        lfb = 1 * b
        lgu = 1
        opti.subject_to(lfb + lgu * u[0, k] >= 0)

        b = -X[3, k] + 20 # to change vmax to something else, modify 20 to the desired value
        lfb = 1 * b
        lgu = -1
        opti.subject_to(lfb + lgu * u[0, k] >= 0)

    # Define the cost function
    cost = 0
    diag_elements_u = [10 * params[16], params[17], 10 * params[18], 10 * params[19]]

    u_ref = np.zeros((Inputnumber + 2, N))
    u_ref[0] = acc_ref * np.ones((1,N))
    normalization_factor = [max(-umin, umax), max(steermax, -steermin), 100, 0.4]
    for i in range(Inputnumber + 2):
        for h in range(N):
            cost += 0.5 * diag_elements_u[i] * ((u[i, h] - u_ref[i][h]) / normalization_factor[i]) ** 2
            cost += 10 ** 8 * (s[0, h]) ** 2
            cost += 10 ** 8 * (s[1, h]) ** 2
            cost += 10 ** 2 * (s[2, h]) ** 2
            cost += 10 ** 8 * (s[3, h]) ** 2

    opti.subject_to(umin <= u[0, :])
    opti.subject_to(u[0, :] <= umax)
    eps3 = 1
    for k in range(0, N):
        V = (X[2, k] - Psi_ref[k]) ** 2
        lfV = 50 * eps3 * V
        lgdelta = 2 * (X[2, k] - Psi_ref[k]) * X[3, k] / (lr + lf)
        opti.subject_to(lfV + lgdelta * u[1, k] - u[3, k] <= 0)

    opti.subject_to(steermin <= u[1, :])
    opti.subject_to(u[1, :] <= steermax)

    opti.subject_to(X[:, 0] == states)  # initialize states
    timespan = [0, dt]

    for h in range(N):  # initial guess
        opti.set_initial(X[:, h], [x0, y0, Psi_ref[0], v0])

    for k in range(N):
        state = []
        Input = []
        for j in range(statesnumber):
            state.append(X[j, k])
        for j in range(0, Inputnumber):
            Input.append(u[j, k])
        state = rk4(timespan, state, Input, 1)
        for j in range(statesnumber):
            opti.subject_to(X[j, k + 1] == state[j])

    opts = {}
    opts['print_time'] = False
    opts['ipopt.print_level'] = False
    opti.solver('ipopt', opts)
    opti.minimize(cost)
    sol = opti.solve()

    if N > 1:
        s_vars = 0
        acc = 1 / Mass * sol.value(u)[0, 0]
        return "solution found", [acc, sol.value(u)[1, 0]], s_vars
    else:
        #print('b3', sol.value(b3_crossing), "lfb3", sol.value(lfb3_crossing), "lgb3u", sol.value(lgb3u_crossing), "cbf_u", -sol.value(lfb3_crossing)/sol.value(lgb3u_crossing))
        if np.any(sol.value(s) > 5):  # or LeftCBF <= 0 or RightCBF <= 0:
            s_vars = 5.1
            acc = -6
            return "No solution found", np.array([acc, 0.0]), s_vars
        else:
            s_vars = 0
            acc = 1 / Mass * sol.value(u[0])
            return "solution found", sol.value(u[:2]), s_vars

def throttle_brake_mapping1(a):
    if a >= 0:
        throttle = a/3.92
        brake = 0
    else:
        throttle = 0
        brake = a/6
    return throttle, brake

def path_plannar(step, envs, cav, vid, agents_dict, agent_dict, command, previous_command, cavs_list, reference_lane_index=0, scenario="crossing", highway_lanebound=None):

    x_centerline = []
    y_centerline = []
    desired_phi = []

    ncav_loc = cav.get_location()
    ncav_waypoint = envs.map.get_waypoint(ncav_loc)
    current_lane_index = ncav_waypoint.lane_id

    admission_flag = IsLaneChangeAdmissible(vid, agents_dict, agent_dict, command, current_lane_index,cavs_list,ncav_waypoint, scenario, highway_lanebound)

    centerlane = []
    for _ in range(0, 10):
        x_centerline.append(ncav_waypoint.transform.location.x)
        y_centerline.append(ncav_waypoint.transform.location.y)
        centerlane.append([ncav_waypoint.transform.location.x, ncav_waypoint.transform.location.y])
        phi = np.deg2rad(ncav_waypoint.transform.rotation.yaw)
        desired_phi.append(phi)
        ncav_waypoint = ncav_waypoint.next(1)[0]

    phi_d = np.arctan2(y_centerline[4] - ncav_loc.y,
                       x_centerline[4] - ncav_loc.x)
    phi_d = (phi_d - 0 + np.pi) % (2 * np.pi) - np.pi + 0
    desired_phi.insert(0, 0.6 * phi_d + 0.4 * desired_phi[0])

    if agent_dict["completion"] == 0:
        if (agent_dict["lane_keeping_count"] > 0) or (not admission_flag) or (command in ["KL_2", "KL_3", "KL_4", "KL_5"]):
            agent_dict["ref_centerlane"] = centerlane
            agent_dict["ref_theta"] = desired_phi
            if command not in ["KL_2", "KL_3", "KL_4", "KL_5"]:
                command = "lane_keeping"
            agent_dict["command"] = command
            if agent_dict["lane_keeping_count"] > 0:
                agent_dict["lane_keeping_count"] -= 1


    if  command != "lane_keeping" and admission_flag:
        print("@@ COMMAND LOGGING: {}, {} - accept new command {}".format(vid, step, command))
        if command == "left_lc":
            reference_lane_index = ncav_waypoint.lane_id + 1
            # print("left lane change command")
            agent_dict["completion"] = 0.001
            agent_dict["lc_starting_time"] = step
            agent_dict["lane_keeping_count"] = 50
            agents_dict["lc_starting_time_2"] = step

        elif command == "right_lc":
            reference_lane_index = ncav_waypoint.lane_id - 1
            # print("right lane change command")
            agent_dict["completion"] = 0.001
            agent_dict["lc_starting_time"] = step
            agent_dict["lane_keeping_count"] = 50
            agents_dict["lc_starting_time_2"] = step

        agent_dict["command"] = command
        if reference_lane_index - current_lane_index == 1:
            future_waypoint = ncav_waypoint.get_left_lane()
            future_waypoint = future_waypoint.next(5)[0]
            delta = 0
        else:
            future_waypoint = ncav_waypoint.get_right_lane()
            future_waypoint = future_waypoint.next(5)[0]
            delta = 0

        p0 = np.array([ncav_loc.x, ncav_loc.y])
        next_ncav_loc = envs.map.get_waypoint(ncav_loc).next(5)[0]
        p1 = np.array([next_ncav_loc.transform.location.x, next_ncav_loc.transform.location.y])
        p2 = np.array([future_waypoint.transform.location.x, future_waypoint.transform.location.y])
        future_waypoint = future_waypoint.next(5)[0]
        p3 = np.array([future_waypoint.transform.location.x, future_waypoint.transform.location.y + delta])
        num_samples = 30
        time_samples = np.linspace(0, 1, num_samples)
        sampled_points = np.array([bezier_curve(t, p0, p1, p2, p3) for t in time_samples])
        for _ in range(1, 20):
            future_waypoint = future_waypoint.next(2)[0]
            new_element = [np.array([future_waypoint.transform.location.x, future_waypoint.transform.location.y])]
            sampled_points = np.append(sampled_points, new_element, axis=0)

        translations_x, translations_y = [sampled_points[0][0] - ncav_loc.x, sampled_points[0][1] - ncav_loc.y]
        centerlane = []
        for k in range(len(sampled_points)):
            centerlane.append([sampled_points[k][0] - translations_x, sampled_points[k][1] - translations_y])
        desired_phi = [
            np.arctan2(centerlane[i + 1][1] - centerlane[i][1], centerlane[i + 1][0] - centerlane[i][0])
            for i in range(len(centerlane) - 1)]


        agent_dict["length_path"] = len(centerlane)
        agent_dict["ref_centerlane"] = np.array(centerlane)
        agent_dict["ref_theta"] = desired_phi


    if (agent_dict["completion"] > 0 and agent_dict["completion"] < 0.8):
        if agent_dict["timeout"] == 0:
            centerlane = agent_dict["ref_centerlane"]
            current_position = np.array([ncav_loc.x, ncav_loc.y])
            lateral_distances = np.abs(centerlane[:, 1] - current_position[1])
            closest_index = np.argmin(lateral_distances)
            agent_dict["completion"] = 0.05 + closest_index / agent_dict["length_path"]
        else:
            print("@@ COMMAND LOGGING: {}, {} - timeout exceeded".format(vid, step))
            agent_dict["completion"] = 0
            agent_dict["timeout"] = 0
            agent_dict["lc_starting_time"] = -1
            agents_dict["lc_starting_time_2"] = -1
            reference_lane_index = current_lane_index
    elif agent_dict["completion"] >= 0.8:
        print("@@ COMMAND LOGGING: {}, {} - successful lane change".format(vid, step))
        agent_dict["lc_starting_time"] = -1
        agents_dict["lc_starting_time_2"] = -1
        agent_dict["completion"] = 0
        reference_lane_index = current_lane_index

    return agent_dict, reference_lane_index, current_lane_index

def IsLaneChangeAdmissible(vid, agents_dict, agent_dict, command, current_lane_index, cavs_list, ncav_waypoint, scenario, highway_lanebound):
    if not highway_lanebound:
        highway_lanebound = [-3, -7]
    x0 = ncav_waypoint.transform.location.x
    if scenario == "highway":
        if agent_dict["completion"] == 0:
            if command == "left_lc" and current_lane_index < highway_lanebound[0]: # and current_lane_index < -4 if we don't want vehicle to drive on to lane -3

                for id in cavs_list:
                    if vid == id:
                        continue
                    if abs(agents_dict[id]['position']['x'] - x0) < 10:
                        if agents_dict[id]['completion'] > 0 and agents_dict[id]["reference_lane_index"] == ncav_waypoint.lane_id + 1:
                            return False
                        if agents_dict[id]['completion'] > 0 and agents_dict[id]["reference_lane_index"] == current_lane_index and agents_dict[id]["current_lane_index"] == ncav_waypoint.lane_id + 1:
                            return False
                return True

            if command == "right_lc" and current_lane_index > highway_lanebound[1]:
                for id in cavs_list:
                    if vid == id:
                        continue
                    if abs(agents_dict[id]['position']['x'] - x0) < 10:
                        if agents_dict[id]['completion'] > 0 and agents_dict[id]["reference_lane_index"] == ncav_waypoint.lane_id - 1:
                            return False
                        if agents_dict[id]['completion'] > 0 and agents_dict[id]["reference_lane_index"] == current_lane_index and agents_dict[id]["current_lane_index"] == ncav_waypoint.lane_id - 1:
                            return False
                return True
        
        return False
    else:
        Junction_entrance = -45
        if agent_dict["completion"] == 0 and agent_dict["position"]['x'] < Junction_entrance:
            if command == "left_lc" and current_lane_index < -3:
                for id in cavs_list:
                    if vid == id:
                        continue
                    if abs(agents_dict[id]['position']['x'] - x0) < 10:
                        if agents_dict[id]['completion'] > 0 and agents_dict[id]["reference_lane_index"] == ncav_waypoint.lane_id + 1:
                            return False
                        if agents_dict[id]['completion'] > 0 and agents_dict[id]["reference_lane_index"] == current_lane_index and agents_dict[id]["current_lane_index"] == ncav_waypoint.lane_id + 1:
                            return False

                return True
            if command == "right_lc" and current_lane_index > -5:
                for id in cavs_list:
                    if vid == id:
                        continue
                    if abs(agents_dict[id]['position']['x'] - x0) < 10:
                        if agents_dict[id]['completion'] > 0 and agents_dict[id]["reference_lane_index"] == ncav_waypoint.lane_id - 1:
                            return False
                        if agents_dict[id]['completion'] > 0 and agents_dict[id]["reference_lane_index"] == current_lane_index and agents_dict[id]["current_lane_index"] == ncav_waypoint.lane_id - 1:
                            return False
                return True
        
        return False

def find_conflicting_cars(envs, agents_dict, cav, agent_dict, ego_id, all_car_info_dict, reference_lane_index,non_cavs_lane_ids, non_cavs_position_list,perturbation_dict, scenario='highway'):
    agent_dict["conflicting_cars"]['ip'] = []
    agent_dict["conflicting_cars"]['ic_1'] = []
    agent_dict["conflicting_cars"]['ic_2'] = []
    agent_dict["MPs"] = []
    conflicting_cars_state_degree1 = {'front': [], 'rear': []}
    conflicting_cars_state_degree2 = []
    conflicting_cars_state_degree3 = {'front': [], 'rear': []}
    preceding_cars_state = []
    non_cavs_list = list(non_cavs_lane_ids.keys())
    ncav_loc = cav.get_location()
    ncav_waypoint = envs.map.get_waypoint(ncav_loc)
    x11 = ncav_waypoint.transform.location.x
    y11 = ncav_waypoint.transform.location.y
    ncav_waypoint = ncav_waypoint.next(1)[0]
    x12 = ncav_waypoint.transform.location.x
    y12 = ncav_waypoint.transform.location.y
    maximum_noise_x = 5
    maximum_noise_v = 5
    if scenario == 'highway':
        for vid, info_dict in all_car_info_dict.items():
            if vid == ego_id:
                states = [all_car_info_dict[vid]['x'], \
                all_car_info_dict[vid]['y'], \
                all_car_info_dict[vid]['phi'], \
                all_car_info_dict[vid]['vel'] / 3.6]
            else:
                if all_car_info_dict[vid]["lane_id"] == all_car_info_dict[ego_id]["lane_id"] or abs(all_car_info_dict[vid]["y"] -  all_car_info_dict[ego_id]["y"]) < 3:
                    if all_car_info_dict[vid]["x"] > all_car_info_dict[ego_id]["x"]:
                        x_ip = all_car_info_dict[vid]["x"] + perturbation_dict[vid][0]
                        y_ip = all_car_info_dict[vid]["y"] + perturbation_dict[vid][1]
                        phi_ip = all_car_info_dict[vid]["phi"] 
                        vel_ip = (all_car_info_dict[vid]["vel"] + np.sqrt(perturbation_dict[vid][2]**2 + perturbation_dict[vid][3]**2)) / 3.6 
                        preceding_cars_state.append([x_ip, y_ip, phi_ip, vel_ip])
                        agent_dict["conflicting_cars"]['ip'].append(vid)
                else:
                    if all_car_info_dict[vid]["lane_id"] == reference_lane_index:
                        if all_car_info_dict[vid]["x"] + maximum_noise_x > all_car_info_dict[ego_id]["x"]:
                            x_ic = all_car_info_dict[vid]["x"] + perturbation_dict[vid][0]
                            y_ic = all_car_info_dict[vid]["y"] + perturbation_dict[vid][1]
                            phi_ic = all_car_info_dict[vid]["phi"]
                            vel_ic = (all_car_info_dict[vid]["vel"] + np.sqrt(perturbation_dict[vid][2]**2 + perturbation_dict[vid][3]**2)) / 3.6 
                            conflicting_cars_state_degree1['front'].append([x_ic, y_ic, phi_ic, vel_ic])
                            agent_dict["conflicting_cars"]['ic_1'].append(vid)
                        else:
                            x_ic = all_car_info_dict[vid]["x"] + perturbation_dict[vid][0]
                            y_ic = all_car_info_dict[vid]["y"] + perturbation_dict[vid][1]
                            phi_ic = all_car_info_dict[vid]["phi"]
                            vel_ic = (all_car_info_dict[vid]["vel"] + np.sqrt(perturbation_dict[vid][2]**2 + perturbation_dict[vid][3]**2)) / 3.6 
                            conflicting_cars_state_degree1['rear'].append([x_ic, y_ic, phi_ic, vel_ic])
                            agent_dict["conflicting_cars"]['ic_1'].append(vid)
                    else:
                        x_ic = all_car_info_dict[vid]["x"] + perturbation_dict[vid][0]
                        y_ic = all_car_info_dict[vid]["y"] + perturbation_dict[vid][1]
                        phi_ic = all_car_info_dict[vid]["phi"]
                        vel_ic = (all_car_info_dict[vid]["vel"] + np.sqrt(perturbation_dict[vid][2]**2 + perturbation_dict[vid][3]**2)) / 3.6 
                        conflicting_cars_state_degree2.append([x_ic, y_ic, phi_ic, vel_ic])
       

    elif scenario == "crossing":
        for vid, info_dict in all_car_info_dict.items():
            if vid == ego_id:
                states = [all_car_info_dict[vid]['x'], \
                all_car_info_dict[vid]['y'], \
                all_car_info_dict[vid]['phi'], \
                all_car_info_dict[vid]['vel'] / 3.6]              
            else:
                if all_car_info_dict[vid]["lane_id"] == all_car_info_dict[ego_id]["lane_id"] or abs(all_car_info_dict[vid]["y"] -  all_car_info_dict[ego_id]["y"]) < 3:
                    if all_car_info_dict[vid]["x"] > all_car_info_dict[ego_id]["x"]:
                        x_ip = all_car_info_dict[vid]["x"] + perturbation_dict[vid][0]
                        y_ip = all_car_info_dict[vid]["y"] + perturbation_dict[vid][1]
                        phi_ip = all_car_info_dict[vid]["phi"]
                        vel_ip = (all_car_info_dict[vid]["vel"] + np.sqrt(perturbation_dict[vid][2]**2 + perturbation_dict[vid][3]**2)) / 3.6 
                        preceding_cars_state.append([x_ip, y_ip, phi_ip, vel_ip])
                        agent_dict["conflicting_cars"]['ip'].append(vid)   
                else:
                    if vid not in non_cavs_list:
                        if all_car_info_dict[vid]["lane_id"] == reference_lane_index:
                            if all_car_info_dict[vid]["x"] > all_car_info_dict[ego_id]["x"]:
                                x_ic = all_car_info_dict[vid]["x"] + perturbation_dict[vid][0]
                                y_ic = all_car_info_dict[vid]["y"] + perturbation_dict[vid][1]
                                phi_ic = all_car_info_dict[vid]["phi"]
                                vel_ic = (all_car_info_dict[vid]["vel"] + np.sqrt(perturbation_dict[vid][2]**2 + perturbation_dict[vid][3]**2)) / 3.6 
                                conflicting_cars_state_degree1['front'].append([x_ic, y_ic, phi_ic, vel_ic])
                                agent_dict["conflicting_cars"]['ic_1'].append(vid)
                            else:
                                x_ic = all_car_info_dict[vid]["x"] + perturbation_dict[vid][0]
                                y_ic = all_car_info_dict[vid]["y"] + perturbation_dict[vid][1]
                                phi_ic = all_car_info_dict[vid]["phi"]
                                vel_ic = (all_car_info_dict[vid]["vel"] + np.sqrt(perturbation_dict[vid][2]**2 + perturbation_dict[vid][3]**2)) / 3.6 
                                conflicting_cars_state_degree1['rear'].append([x_ic, y_ic, phi_ic, vel_ic])
                                agent_dict["conflicting_cars"]['ic_1'].append(vid)
                                         
                    else:
                        x21 = non_cavs_position_list[vid]['pre_x']
                        y21 = non_cavs_position_list[vid]['pre_y']
                        x22 =  all_car_info_dict[vid]["x"] 
                        y22 = all_car_info_dict[vid]["y"] 
                        determinant = find_intersection(x11, y11, x12, y12, x21, y21, x22, y22)
                        if determinant is not None:
                            MP_x = determinant[0]
                            MP_y = determinant[1]
                            x_ic = all_car_info_dict[vid]["x"] + perturbation_dict[vid][0]
                            y_ic = all_car_info_dict[vid]["y"] + + perturbation_dict[vid][1]
                            phi_ic = all_car_info_dict[vid]["phi"]
                            vel_ic = (all_car_info_dict[vid]["vel"] + np.sqrt(perturbation_dict[vid][2]**2 + perturbation_dict[vid][3]**2)) / 3.6 
                            agent_dict["MPs"].append([MP_x, MP_y])
                            conflicting_cars_state_degree2.append([x_ic, y_ic, phi_ic, vel_ic])
                            agent_dict["conflicting_cars"]['ic_2'].append(vid)
                            conflicting_cars_state_degree3["front"].append([x_ic, y_ic, phi_ic, vel_ic])
                            conflicting_cars_state_degree3["rear"].append([x_ic, y_ic, phi_ic, vel_ic])

    return agent_dict, states , preceding_cars_state, conflicting_cars_state_degree1, conflicting_cars_state_degree2, agent_dict["MPs"],conflicting_cars_state_degree3



def find_intersection(x11, y11, x12, y12, x21, y21, x22, y22):
    """
    Finds the intersection point of two lines defined by the points (x11, y11) -> (x12, y12) and (x21, y21) -> (x22, y22).

    Parameters:
        x11, y11: coordinates of the first point on the first line (Car 1)
        x12, y12: coordinates of the second point on the first line (Car 1)
        x21, y21: coordinates of the first point on the second line (Car 2)
        x22, y22: coordinates of the second point on the second line (Car 2)

    Returns:
        (x, y): coordinates of the intersection point or None if the lines are parallel
    """
    # Calculate the coefficients of the lines
    A1 = y12 - y11
    B1 = x11 - x12
    C1 = A1 * x11 + B1 * y11

    A2 = y22 - y21
    B2 = x21 - x22
    C2 = A2 * x21 + B2 * y21

    # Calculate the determinant
    determinant = A1 * B2 - A2 * B1

    if determinant == 0:
        # Lines are parallel, no intersection
        return None
    else:
        # Calculate the intersection point
        x = (B2 * C1 - B1 * C2) / determinant
        y = (A1 * C2 - A2 * C1) / determinant
        return x, y


def crossing_high_level_controller(agent_dict, acc_ref, states, preceding_cars_state, conflicting_cars_state_deg1, conflicting_cars_state_deg2, vid, vid_select, current_lane_index,non_cavs_lane_ids, MPs,conflicting_cars_state_deg3):
    
    Psi_ref = agent_dict["ref_theta"]
    centerlane = agent_dict["ref_centerlane"]
    closest_index = round(agent_dict["completion"] * agent_dict["length_path"])
    closest_index = min(closest_index, len(Psi_ref) - 8)
    Psi_ref = Psi_ref[closest_index:]

    veh_len = 1
    dt = 0.1
    eps = 1
    statesnumber = 4
    Inputnumber = 2
    N = 1

    params = 20 * [0.5]
    x0, y0, psi0, v0 = states
    opti = cd.Opti()
    umin = -0.6 * 9.81 * Mass
    umax = 0.4 * 9.81 * Mass

    steermin = -0.6 * (35 - v0)/35
    steermax = 0.6 * (35 - v0)/35

    X = opti.variable(statesnumber, N + 1)
    u = opti.variable(Inputnumber + 2, N)
    s = opti.variable(4, N)

    maximum_noise_x = 5
    maximum_noise_v = 5
    b3_crossing  =[]
    lfb3_crossing = []
    lgb3u_crossing = []


    if current_lane_index != agent_dict["reference_lane_index"]:
        for vehicle_states in conflicting_cars_state_deg1['front']:
            curr_states = vehicle_states
            for k in range(0, N):
                a_x = 1
                a_y = 0.8
                k1 = 10
                x_ic, y_ic, psi_ic, v_ic = curr_states
                y_ic = 0.5 * y_ic + 0.5*y0 
                if x_ic < x0:
                    x_ic = min(x_ic + maximum_noise_x, x0)
                    v_ic_2 = v_ic + maximum_noise_v
                else:
                    x_ic = max(x_ic - maximum_noise_x, x0)
                    v_ic_2 = v_ic - maximum_noise_v

                if y_ic < y0:
                    v_ic_1 = v_ic + maximum_noise_v
                else:
                    v_ic_1 = v_ic - maximum_noise_v

                b3 = -veh_len ** 2 + (X[1, k] - y_ic) ** 2 / (a_y ** 2 * (eps + X[3, k]) ** 2) + (X[0, k] - x_ic) ** 2 / (a_x ** 2 * (eps + X[3, k]) ** 2)

                lfb3 = (k1 * (-veh_len ** 2 + (X[1, k] - y_ic) ** 2 / (a_y ** 2 * (eps + X[3, k]) ** 2) + (X[0, k] - x_ic) ** 2 / (a_x ** 2 * (eps + X[3, k]) ** 2)) +
                        2 * (a_x ** 2 * (X[1, k] - y_ic) * ( X[3, k] * cd.sin(beta + X[2, k]) - v_ic_1 * cd.sin(beta + psi_ic)) +
                                a_y ** 2 * (X[0, k] - x_ic) * (X[3, k] * cd.cos(beta + X[2, k]) - v_ic_2 * cd.cos(beta + psi_ic))) / (a_x ** 2 * a_y ** 2 * (eps + X[3, k]) ** 2))
                lgb3delta = 0
                lgb3u = (-2 * (X[1, k] - y_ic) ** 2 / (a_y ** 2 * (eps + X[3, k]) ** 3) - 2 * (X[0, k] - x_ic) ** 2 / (a_x ** 2 * (eps + X[3, k]) ** 3)) / Mass

                opti.subject_to(lfb3 + lgb3u * u[0, k] + lgb3delta * u[1, k] + s[2, k] >= 0)
                curr_states = motion(dynamics, curr_states, [0, 0], [0, dt], dt)
                
        for vehicle_states in conflicting_cars_state_deg1['rear']:
            curr_states = vehicle_states
            for k in range(0, N):
                a_x = 1
                a_y = 0.8
                k1 = 10
                x_ic, y_ic, psi_ic, v_ic = curr_states
                y_ic = 0.5 * y_ic + 0.5*y0
                if x_ic < x0:
                    x_ic = min(x_ic + maximum_noise_x, x0)
                    v_ic_2 = v_ic + maximum_noise_v
                else:
                    x_ic = max(x_ic - maximum_noise_x, x0)
                    v_ic_2 = v_ic - maximum_noise_v

                if y_ic < y0:
                    v_ic_1 = v_ic + maximum_noise_v
                else:
                    v_ic_1 = v_ic - maximum_noise_v

                b3 = -veh_len ** 2 + (-X[1, k] + y_ic) ** 2 / (a_y ** 2 * (eps + X[3, k]) ** 2) + (-X[0, k] + x_ic) ** 2 / (a_x ** 2 * (eps + X[3, k]) ** 2)

                lfb3 = (k1 * (-veh_len ** 2 + (-X[1, k] + y_ic) ** 2 / (a_y ** 2 * (eps + X[3, k]) ** 2) + (-X[0, k] + x_ic) ** 2 / (a_x ** 2 * (eps + X[3, k]) ** 2)) +
                        2 * (a_x ** 2 * (-X[1, k] + y_ic) * ( -X[3, k] * cd.sin(beta + X[2, k]) + v_ic_1 * cd.sin(beta + psi_ic)) +
                                a_y ** 2 * (-X[0, k] + x_ic) * (-X[3, k] * cd.cos(beta + X[2, k]) + v_ic_2 * cd.cos(beta + psi_ic))) / (a_x ** 2 * a_y ** 2 * (eps + X[3, k]) ** 2))
                lgb3delta = 0
                lgb3u = (2 * (X[1, k] - y_ic) ** 2 / (a_y ** 2 * (eps + X[3, k]) ** 3) + 2 * (X[0, k] - x_ic) ** 2 / (a_x ** 2 * (eps + X[3, k]) ** 3)) / Mass

                opti.subject_to(lfb3 + lgb3u * u[0, k] + lgb3delta * u[1, k] + s[2, k] >= 0)
                curr_states = motion(dynamics, curr_states, [0, 0], [0, dt], dt)

    for vehicle_states in preceding_cars_state:
        curr_states = vehicle_states
        for k in range(0, N):
            a_x = 1
            a_y = 0.02
            k1 = 10
            x_ip, y_ip, psi_ip, v_ip = curr_states
            v_ip = v_ip - maximum_noise_v
            x_ip = x_ip - maximum_noise_x
            y_ip = states[1]
            b3 = -veh_len ** 2 + (X[1, k] - y_ip) ** 2 / (a_y ** 2 * (eps + X[3, k]) ** 2) + (X[0, k] - x_ip) ** 2 / (
                        a_x ** 2 * (eps + X[3, k]) ** 2)
            lfb3 = (k1 * b3
                    + 2 * (a_x ** 2 * (X[1, k] - y_ip) * (
                    X[3, k] * cd.sin(beta + X[2, k]) - v_ip * cd.sin(beta + psi_ip)) + a_y ** 2 * (X[0, k] - x_ip) * (X[3, k] * cd.cos(beta + X[2, k]) - v_ip * cd.cos(beta + psi_ip))) / (a_x ** 2 * a_y ** 2 * (eps + X[3, k]) ** 2))
            lgb3delta = 0
            lgb3u = (-2 * (X[1, k] - y_ip) ** 2 / (a_y ** 2 * (eps + X[3, k]) ** 3) - 2 * (X[0, k] - x_ip) ** 2 / (a_x ** 2 * (eps + X[3, k]) ** 3)) / Mass
            opti.subject_to(lfb3 + lgb3u * u[0, k] + lgb3delta * u[1, k] + s[0, k] >= 0)
            curr_states = motion(dynamics, curr_states, [0, 0], [0, dt], dt)
            if k == 0:
                b3_ip = b3
                lfb3_ip = lfb3
                lgb3u_ip = lgb3u
                lgb3delta_ip = lgb3delta
                cbf_ip = lfb3_ip + lgb3u_ip * u[0, k] + lgb3delta_ip * u[1, k]

    for index, vehicle_states in enumerate(conflicting_cars_state_deg3['front']):
        curr_states = vehicle_states
        x_ic, y_ic, psi_ic, v_ic = curr_states
        sequence_parameter = 1.2
        mp = MPs[index]
        d1 = mp[0] - x0
        varphi = 1.2
        length = [mp[0] - agent_dict["initial_position"]['x']]
        if x_ic > 0:
            y_ic = y_ic - maximum_noise_x
            v_ic = v_ic - maximum_noise_v
            d2 = y_ic - mp[1]
        else:
            y_ic = y_ic + maximum_noise_x
            v_ic = v_ic - maximum_noise_v
            d2 = mp[1] - y_ic
        if d1 - d2  > -sequence_parameter * v_ic - 6.74:
            if mp[0] - x0 >= 0:
                for k in range(0, N): 
                    d1 = mp[0] - X[0,k]
                    if non_cavs_lane_ids[agent_dict["conflicting_cars"]['ic_2'][index]] == -2:
                        b3 = d1 - d2 - varphi/length[0]*(X[0,k] - agent_dict["initial_position"]['x']) * X[3,k] - veh_len
                        lfb3 =-X[3,k]*cd.cos(beta + X[2,k]) - v_ic*cd.sin(beta + psi_ic) - varphi/length[0]*(X[3,k]*cd.cos(beta + X[2,k])) * X[3,k] + 1*(b3)
                    else:                    
                        b3 = d1 - d2 - varphi/length[0]*(X[0,k] - agent_dict["initial_position"]['x']) * X[3,k] - veh_len
                        lfb3 =-X[3,k]*cd.cos(beta + X[2,k]) + v_ic*cd.sin(beta + psi_ic) - varphi/length[0]*(X[3,k]*cd.cos(beta + X[2,k])) * X[3,k] + 1*(b3)
                
                    lgb3delta = 0
                    lgb3u = -varphi/length[0]*(X[0,k] - agent_dict["initial_position"]['x'])
                
                    opti.subject_to(lfb3 + lgb3u * u[0, k] + lgb3delta * u[1, k] + s[2, k] >= 0)
                    curr_states = motion(dynamics, curr_states, [0, 0], [0, dt], dt)

    for k in range(0, N):
        b = X[3, k] - 0
        lfb = 1 * b
        lgu = 1
        opti.subject_to(lfb + lgu * u[0, k] >= 0)

        b = -X[3, k] + 20 # to change vmax to something else, modify 20 to the desired value
        lfb = 1 * b
        lgu = -1
        opti.subject_to(lfb + lgu * u[0, k] >= 0)

    # Define the cost function
    cost = 0
    diag_elements_u = [10 * params[16], params[17], 10 * params[18], 10 * params[19]]

    u_ref = np.zeros((Inputnumber + 2, N))
    u_ref[0] = acc_ref * np.ones((1,N))
    normalization_factor = [max(-umin, umax), max(steermax, -steermin), 100, 0.4]
    for i in range(Inputnumber + 2):
        for h in range(N):
            cost += 0.5 * diag_elements_u[i] * ((u[i, h] - u_ref[i][h]) / normalization_factor[i]) ** 2
            cost += 10 ** 8 * (s[0, h]) ** 2
            cost += 10 ** 8 * (s[1, h]) ** 2
            cost += 10 ** 2 * (s[2, h]) ** 2
            cost += 10 ** 8 * (s[3, h]) ** 2

    opti.subject_to(umin <= u[0, :])
    opti.subject_to(u[0, :] <= umax)
    eps3 = 1
    for k in range(0, N):
        V = (X[2, k] - Psi_ref[k]) ** 2
        lfV = 50 * eps3 * V
        lgdelta = 2 * (X[2, k] - Psi_ref[k]) * X[3, k] / (lr + lf)
        opti.subject_to(lfV + lgdelta * u[1, k] - u[3, k] <= 0)

    opti.subject_to(steermin <= u[1, :])
    opti.subject_to(u[1, :] <= steermax)

    opti.subject_to(X[:, 0] == states)  # initialize states
    timespan = [0, dt]

    for h in range(N):  # initial guess
        opti.set_initial(X[:, h], [x0, y0, Psi_ref[0], v0])

    for k in range(N):
        state = []
        Input = []
        for j in range(statesnumber):
            state.append(X[j, k])
        for j in range(0, Inputnumber):
            Input.append(u[j, k])
        state = rk4(timespan, state, Input, 1)
        for j in range(statesnumber):
            opti.subject_to(X[j, k + 1] == state[j])

    opts = {}
    opts['print_time'] = False
    opts['ipopt.print_level'] = False
    opti.solver('ipopt', opts)
    opti.minimize(cost)
    sol = opti.solve()

    if N > 1:
        s_vars = 0
        acc = 1 / Mass * sol.value(u)[0, 0]
        return "solution found", [acc, sol.value(u)[1, 0]], s_vars
    else:
        #print('b3', sol.value(b3_crossing), "lfb3", sol.value(lfb3_crossing), "lgb3u", sol.value(lgb3u_crossing), "cbf_u", -sol.value(lfb3_crossing)/sol.value(lgb3u_crossing))
        if np.any(sol.value(s) > 5):  # or LeftCBF <= 0 or RightCBF <= 0:
            s_vars = 5.1
            acc = -6
            return "No solution found", np.array([acc, 0.0]), s_vars
        else:
            s_vars = 0
            acc = 1 / Mass * sol.value(u[0])
            return "solution found", sol.value(u[:2]), s_vars


def high_level_lane_changing(envs, cav, agents_dict, all_car_info_dict, non_cavs_list, ego_id , b_safe, p, acceleration_thereshold, perturbation_dict, scenario, highway_lanebound):

    if not highway_lanebound:
        highway_lanebound = [-3, -7]

    commands = ["right_lc", "left_lc"]    
    ncav_loc = cav.get_location()
    ncav_waypoint = envs.map.get_waypoint(ncav_loc)
    states = [all_car_info_dict[ego_id]['x'],all_car_info_dict[ego_id]['y'], all_car_info_dict[ego_id]['phi'],all_car_info_dict[ego_id]['vel']/3.6]
    command_result = "lane_keeping"
    max_lane_changing_motivation = -100
    current_lane_index = ncav_waypoint.lane_id

    for command in commands:
        if scenario == "highway":
            if command == "left_lc" and current_lane_index >= highway_lanebound[0]:
                continue
            if command == "right_lc" and current_lane_index <= highway_lanebound[1]:
                continue
        a_c = agents_dict[ego_id]["acceleration"]
        tilde_a_c = 3.92
        tilde_a_n = 0
        a_n = 0
        tilde_a_o = 0
        a_o = 0
        if command == "left_lc":
            reference_lane_index = ncav_waypoint.lane_id + 1
        else:
            reference_lane_index = ncav_waypoint.lane_id - 1
        
        for vid, _ in all_car_info_dict.items():
            if vid == ego_id:
                continue
            else:
                if all_car_info_dict[vid]["lane_id"] == reference_lane_index:
                    if all_car_info_dict[vid]["x"] > all_car_info_dict[ego_id]["x"]:
                        x_ic = all_car_info_dict[vid]["x"] + perturbation_dict[vid][0]
                        y_ic = all_car_info_dict[vid]["y"] + perturbation_dict[vid][1]
                        phi_ic = all_car_info_dict[vid]["phi"]
                        vel_ic = (all_car_info_dict[vid]["vel"] + np.sqrt(perturbation_dict[vid][2]**2 + perturbation_dict[vid][3]**2)) / 3.6 
                        tilde_a_c = (vel_ic - states[3]) / 0.5
                    else:
                        x_ic = all_car_info_dict[vid]["x"] + perturbation_dict[vid][0]
                        y_ic = all_car_info_dict[vid]["y"] + perturbation_dict[vid][1]
                        phi_ic = all_car_info_dict[vid]["phi"]
                        vel_ic = (all_car_info_dict[vid]["vel"] + np.sqrt(perturbation_dict[vid][2]**2 + perturbation_dict[vid][3]**2)) / 3.6 
                        new_vel_ic = (states[0] - x_ic)/1.4
                        if vid in non_cavs_list:
                            a_n = 0
                            tilde_a_n = 0
                        else:
                            a_n = agents_dict[vid]["acceleration"]
                            tilde_a_n = max(-5.88,min((new_vel_ic - vel_ic) / 0.5,3.92))

        if tilde_a_n >= -b_safe:
            lane_changing_motivation = tilde_a_c - a_c + p*(tilde_a_n - a_n + tilde_a_o - a_o) 
            if lane_changing_motivation >= acceleration_thereshold:
                if lane_changing_motivation > max_lane_changing_motivation:
                    max_lane_changing_motivation = lane_changing_motivation
                    command_result = command
    
    return command_result

