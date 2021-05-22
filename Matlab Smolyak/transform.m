function [y] = transform(f, xx, u, a)
    point = (xx ./ 2) + 0.5 
    y = f(point,u,a)
end