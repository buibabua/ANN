function out = gra_amp_costfunction(wb)
global k x_rpt y_rpt z_rpt input_rpt x_mesh y_mesh 

[W,b] = wb_convert(wb,k);

%----------------- Interpolate Z of rpt data --------------------
Zgrid_rpt = griddata(x_rpt,y_rpt,z_rpt,x_mesh,y_mesh);

%----------------- Calculate amp of gradient grid for rpt data -------------
[FX_rpt,FY_rpt] = gradient(Zgrid_rpt);

for row = 1:size(FX_rpt,1)
    for col = 1:size(FX_rpt,2)
        if (isnan(FX_rpt(row,col))== 1 | isnan(FY_rpt(row,col))== 1);
            graamp_rpt(row,col) = 0;
        else
            graamp_rpt(row,col) = sqrt(FX_rpt(row,col)^2 +  FY_rpt(row,col)^2);
        end
    end
end

%----------------- Calculate gradient grid for model data -----------------------
z_model = mlpmm(input_rpt,k,W,b);
Zgrid_model = griddata(x_rpt,y_rpt,z_model',x_mesh,y_mesh);
[FX_model,FY_model] = gradient(Zgrid_model);

%----------------- Calculate direction grid for model data -------------------------
for row = 1:size(FX_model,1)
    for col = 1:size(FX_model,2)
        if (isnan(FX_model(row,col))== 1 | isnan(FY_model(row,col))== 1);
            graamp_model(row,col) = 0;
        else
            graamp_model(row,col) = sqrt(FX_model(row,col)^2 +  FY_model(row,col)^2);
        end
    end
end

%----------------- Calculate direction mse -------------------------
for row = 1:size(FX_model,1)
    for col = 1:size(FX_model,2)
        if (isnan(graamp_rpt(row,col))== 1 | isnan(graamp_model(row,col))== 1);
            graamp_model(row,col) = 0;
            graamp_rpt(row,col) = 0;
        end
    end
end
out = immse(graamp_model,graamp_rpt);
save('fval_graamp.txt','out','-append','-ascii');
end