import smolyak
def plot_2d_sparse_grid():
    a = [2,2]
    u = [.5,.5]
    f = smolyak.Hyper_Plane(a,u,boundary_transform=True)
    d=2
    def plot(l):
        q = smolyak.q(l,f)
        f.record_evaluations_to_plot()
        smolyak_integral = q.integrate()
        f.plot_evaluated_points("eval_"+str(l)+".png", f"Smolyak Sparse Grid, Level={l}")
    for l in range(0, 5):
        plot(l)

def plot_3d_functions():
    a = [5,5]
    u = [.5,.5]
    functions = [
        smolyak.Continuous_Function(a,u), 
        smolyak.Gaussian_Function(a,u),
        smolyak.Oscillatory_Function(a,u),
        smolyak.Discontinuous_Function(a,u)
    ]

    for f in functions:
        f.plot()

if __name__ == "__main__":
    plot_2d_sparse_grid()
    plot_3d_functions()