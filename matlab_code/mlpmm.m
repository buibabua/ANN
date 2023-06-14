function output = mlpmm(X,k,W,b)
global input_ps target_ps
T = numel(k) - 1;
Z = cell(T,1);
Z{1} = mapminmax('apply',X,input_ps);

if T >=2
    for t = 1:T-1
        Z{t+1} = tansig(W{t}*Z{t}+b{t});
    end
end
output = tansig(W{T}*Z{T}+b{T});
output = mapminmax('reverse',output,target_ps);
end


