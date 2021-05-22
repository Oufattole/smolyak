import smolyak
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dimensions = [3, 5, 7, 10, 15]
runs = 5
levels = [1,2,3,4,5,6,7,8, 9, 10, 11, 12,13,14,15,16]
# levels = [8,15,20]
functions = [
    smolyak.Continuous_Function, 
    smolyak.Gaussian_Function,
    smolyak.Oscillatory_Function,
    smolyak.Discontinuous_Function
]
ids = [
    "cont",
    "gaus",
    "oscil",
    "disc",
    ]
fname_to_id = {function.name(): id for function, id in zip(functions, ids)}
def read_smolyak(d):
    smolyak_integral_steps = {}
    smolyak_function = {}
    df = pd.read_csv("smolyak_data/"+str(d)+"_smolyak.csv")
    #for dim, level, function_name, r, steps, integral, a, u  in df.iterrows():
        # smolyak_data.setdefault((dim,level,function_name), [])
        # smolyak_data[(dim,level,function_name)].append([r, steps, integral, a, u])
    for a, b  in df.iterrows():
        # print("------------------")
        
        dim, level, function_name, r, steps, integral = b.tolist()[:6]
        au = b.tolist()[6:]
        a = au[:-len(au)//2]
        u = au[-len(au)//2:]
        assert(len(a) +len(u) + 6 == len(b.tolist()))
        assert(dim==d)
        # if function_name == "oscil":
        #     print(b.tolist())
        #     print(a)
        #     print(u)
        #     raise()

        smolyak_integral_steps.setdefault((r,function_name), [])
        smolyak_integral_steps[(r, function_name)].append((level, integral, steps))

        smolyak_function.setdefault((r,function_name), [a, u])
        assert(smolyak_function[(r,function_name)] == [a,u])
    for key in smolyak_integral_steps.keys():
        smolyak_integral_steps[key].sort()
    return smolyak_integral_steps, smolyak_function
        
def cuba_level_to_points(level):
    return 2**level
def mc_level_to_points(level):
    return 2**level
def smolyak_level_to_points(level):
    return 2**level #fix
def test_cubature(f, d, level, ground_truth):
    print(f)
    print(d)
    
    cuba_max_points = cuba_level_to_points(level)
    print(cuba_max_points)
    f.reset_record_count()
    cuba = smolyak.adaptive_cubature(f,n_max=cuba_max_points)
    cuba_points = f.get_count()
    print(f"cuba count:{cuba_points}")
    print("finish cuba")
    mc_max_points = cuba_max_points
    f.reset_record_count()
    mc = smolyak.mc_integrate(f,n_max=mc_max_points)
    mc_points = f.get_count()
    print(f"mc_count:{mc_points}")
    print("finish mc")

    return abs((cuba - ground_truth)/ground_truth), abs((mc - ground_truth)/ground_truth),  cuba_points, mc_points
def test_cubatures(func, d, levels, runs):
    smolyak_integral_steps, smolyak_function = read_smolyak(d)

    smolyak_rel_error = {}#[0 for i in range(len(smolyak_integral_steps))]
    # smolyak_points = [0 for i in range(len(smolyak_integral_steps))]

    cuba_rel_error = {}
    c_freq = {}
    mc_rel_error = {}
    mc_freq ={}
    s_freq = {}

    cuba_points, mc_points, smolyak_points = [], [], []

    function_id = fname_to_id[func.name()]
    for run in range(1,runs+1):
        #replace with smolyak's u and a
        a, u = smolyak_function[(run, function_id)]
        smolyak_results = smolyak_integral_steps[(run, function_id)]
        # u = np.array
        # a = np.random.rand(d)*2+1
        f = func(a,u)
        # print(f"a:{a}")
        # print(f"u:{u}")
        error = 1e-4
        ground_truth = None
        if func == smolyak.Discontinuous_Function:
            ground_truth = smolyak.adaptive_cubature(f, abs_err=error, discontinuous=u)
        else:
            ground_truth = smolyak.mc_integrate(f, rel_err=error)
        # print(f"gt:{ground_truth}")
        for i in range(len(levels)):
            level = levels[i]
            cuba_error, mc_error, cuba_points, mc_points = test_cubature(f,d,level,ground_truth)

            cuba_rel_error.setdefault(cuba_points, 0)
            c_freq.setdefault(cuba_points, 0)
            cuba_rel_error[cuba_points] += cuba_error
            c_freq[cuba_points] += 1

            mc_rel_error.setdefault(mc_points, 0)
            mc_freq.setdefault(mc_points, 0)
            mc_rel_error[mc_points] += mc_error
            mc_freq[mc_points] += 1

        for i in range(len(smolyak_results)):
            level, integral, steps = smolyak_results[i]
            # print(integral)
            # raise()
            smolyak_rel_error.setdefault(steps,0)
            smolyak_rel_error[steps] += abs((integral - ground_truth)/ground_truth) #[i] += abs((integral - ground_truth)/ground_truth)
            s_freq.setdefault(steps,0)
            s_freq[steps] += 1
            # smolyak_points[i] = steps
    for key in cuba_rel_error.keys():
        cuba_rel_error[key] /= c_freq[key]
    
    for key in mc_rel_error.keys():
        mc_rel_error[key] /= mc_freq[key]

    for key in smolyak_rel_error.keys():
        smolyak_rel_error[key] /= s_freq[key]

    return cuba_rel_error, mc_rel_error, smolyak_rel_error

def get_smolyak_y(d,l,func):
    #get from csv
    return 0
def plot(d, func_name, cuba, mc, smolyak, levels):
    def get_rel_error_points(rel_error_dict):
        rel_errors = []
        points = []
        for key in sorted(rel_error_dict.keys()):
            rel_errors.append(rel_error_dict[key])
            points.append(key)
        return rel_errors, points

    cuba_rel_error, cuba_points = get_rel_error_points(cuba)
    mc_rel_error, mc_points = get_rel_error_points(mc)
    smolyak_rel_error, smolyak_points = get_rel_error_points(smolyak)

    # smolyak_points = [smolyak_level_to_points(level) for level in levels]

    x_1 = np.log(cuba_points)
    x_2 = np.log(mc_points)
    x_3 = np.log(smolyak_points)
    x_3 = [each for each in x_3 if each >= min(min(x_1), min(x_2))]

    y_1 = cuba_rel_error
    y_2 = mc_rel_error
    y_3 = smolyak_rel_error[- len(x_3):]

    # y_1 = np.log(cuba)
    # y_2 = np.log(mc)
    # y_3 = np.log(smolyak)
    # print('wth')
    # print(x_1)
    # print(y_1)
    

    title = f"Cubature Relative Error, {d} Dimensions"
    x_label = "Log Number of Function Evaluations"
    y_label = "Relative Error"

    l1, = plt.plot(x_1, y_1, 'go')
    l2, = plt.plot(x_2, y_2, 'bo')
    l3, = plt.plot(x_3, y_3, 'ro')

    plt.legend((l1, l2, l3),('Adaptive Cubature','Monte Carlos Cubature', "Smolyak Cubature"), loc = 'best')
    # plt.legend((l1, l2),('Adaptive Cubature', "Monte Carlos Cubature"), loc = 'best')

    plt.ylim(top=max(max(y_1),max(y_2),max(y_3)))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(f"figures/{func_name}_{d}.png")
    plt.clf()

    print(cuba)
    print(mc)
    print(smolyak)
    # raise("success")

def plot_comparisons():
    for func in functions:
        func_name = func.name()
        for d in dimensions:
            if d>=10 and func is smolyak.Discontinuous_Function:
                continue
            print(f"----------------------Commence dimension {d}-----------------------------")
            ls = levels
            if d == 15:
                ls = levels[:-1]
            cuba_rel_error, mc_rel_error, smolyak_rel_error = test_cubatures(func, d, ls, runs)
            
            plot(d, func_name, cuba_rel_error, mc_rel_error, smolyak_rel_error, ls)
# x, y = read_smolyak(3)
# print(x.keys())
plot_comparisons()  
