function [bpt_value] = boundary_point_transform(f, xx, u, a)
% Ths function transforms f that from [0,1]^d to [-1,1]^d
%while keeping the same integral for []
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% transforms function f to function fh and evaluates the point xx for fh
% such that the integral of f on [0,1]^d equals the integral of fh on
% [-1,1]^d
% INPUTS:
% f = function with inputs (xx, u, a)
% xx = [x1, x2, ..., xd] %  this is the point to evaluate at
% u = [u1, u2, ..., ud] (optional), with default value % 
%     [0.5, 0.5, ..., 0.5] % u defines the function f
% a = [a1, a2, ..., ad] (optional), with default value [5, 5, ..., 5]
% a also defines the function f
%RETURNS:
% xx evaluated on fh
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    point = (xx ./ 2) + 0.5 ;
    d = length(xx);
    bpt_value = f(point,u,a)/(2^(d));
end