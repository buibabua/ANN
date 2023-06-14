function out = mat_rpt_costfunction(wb)
global input_rpt target_rpt k
[W,b] = wb_convert(wb,k);

mlp_output = mlpmm(input_rpt,k,W,b);
out = immse(mlp_output,target_rpt);
save('fval_mat_rpt.txt','out','-append','-ascii');
end
