function out = gra_dir_costfunction(wb)
global k x_rpt y_rpt z_rpt input_rpt x_mesh y_mesh 

[W,b] = wb_convert(wb,k);

%----------------- Interpolate Z of rpt data --------------------
Zgrid_rpt = griddata(x_rpt,y_rpt,z_rpt,x_mesh,y_mesh);

%----------------- Calculate amp of gradient grid for rpt data -------------
[FX_rpt,FY_rpt] = gradient(Zgrid_rpt);

for row = 1:size(FX_rpt,1)
    for col = 1:size(FX_rpt,2)
        if (isnan(FX_rpt(row,col))== 1 | isnan(FY_rpt(row,col))== 1);
            direc_rpt(row,col) = 0;
        else
            direc_rpt(row,col) = atan(FX_rpt(row,col)/FY_rpt(row,col));
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
            direc_model(row,col) = 0;
        else
            direc_model(row,col) = atan(FX_model(row,col)/FY_model(row,col));
        end
    end
end

%----------------- Calculate direction mse -------------------------
for row = 1:size(FX_model,1)
    for col = 1:size(FX_model,2)
        if (isnan(direc_rpt(row,col))== 1 | isnan(direc_model(row,col))== 1);
            direc_model(row,col) = 0;
            direc_rpt(row,col) = 0;
        end
    end
end
out = immse(direc_model,direc_rpt);
save('fval_gradir.txt','out','-append','-ascii');
end