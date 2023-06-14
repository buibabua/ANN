function out = mat_costfunction(wb)
global input_well target_well k
[W,b] = wb_convert(wb,k);

mlp_output = mlpmm(input_well,k,W,b);
out = immse(mlp_output,target_well);
save('fval_mat.txt','out','-append','-ascii');
end
