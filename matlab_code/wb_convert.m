function [W,b] = wb_convert(wb,k)
n = 1;
b = cell(1,numel(k)-1);
W = cell(1,numel(k)-1);
for i=1:(numel(k)-1) % i = i layer 
    b{i} = wb(n:(n + k(i+1) - 1));
    n = n + k(i+1);
    W{i} = wb(n:(n + k(i)*k(i+1) - 1));
    n = n + k(i)*k(i+1);
    W{i} = reshape(W{i},[k(i+1),k(i)]);
end
end