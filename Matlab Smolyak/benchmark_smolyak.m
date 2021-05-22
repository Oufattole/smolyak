dimensions = [3, 5, 7, 10, 15];
levels = {[1,11], [1,8], [1,7], [1,5], [1,4]}; %[8, 9, 10, 11, 12,13,14,15,16]
function_names = ["cont","gaus","disc","oscil"] ;
runs = 5;

for d = 1:length(dimensions)
    interval = levels{d};
    results = table;
    dim=dimensions(d);
    display(dim)
    for f = 1:length(function_names)
        if f == 1
            fun = @cont;
        elseif f == 2
            fun = @gaussian;
        elseif f ==3
            fun = @disc;
        else
            assert(f==4);
            fun = @oscil;
        end
        for r = 1:runs
            u = rand(1,dim);
            a = (rand(1,dim) .* 0.25 + .25); % ranges from 1 to 3
            
            smolyak_fun = @(point) boundary_point_transform(fun, point, u, a);
            for l = interval(1):interval(2)
                k = l;
                
                [integral, steps] = smolyak_integrate(smolyak_fun, dim, k);
                temp_table = table(dim, k, function_names(f), r, steps, integral, a, u);
                results = [results;temp_table];
            end
        end
    end
    store = strcat(num2str(dim) + "_smolyak.csv")
    writetable(results,store);
end



