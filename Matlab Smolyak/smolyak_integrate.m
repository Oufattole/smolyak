function [y,e] = smolyak_integrate(fun, dim,k)
%*****************************************************************************
% nwSpGr_demo: test & demo for nwspgr
% integrate function using sparse grids and simulation
%*****************************************************************************
    [x,w] = spquad(dim, k, repmat([-1,1]',1,dim));
    numnodes = length(w);
    integral = 0;
    for i = 1 : length(w)
        point = x(i,:);
        weight = w(i);
        point_value = fun(point);
        integral = integral+ point_value  * weight;
    end
    y = integral;
    e = numnodes;
end

