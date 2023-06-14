clear all;
close all;
delete 'fval_gradir.txt' 'fval_graamp.txt' 'fval_mat.txt' 'fval_mat_rpt.txt'

global n k w_gra_dir w_gra_amp w_mat w_mat_rpt T W0 b0 wb0 x_grid y_grid
global input_ps target_ps rpt_data x_rpt y_rpt z_rpt input_rpt target_rpt
global well_data input_well target_well x_mesh y_mesh mesh_data 

well_data = load('./data.txt','-ascii');
input_well = well_data(:,1:2)'; % Zp Vp/Vs
target_well = well_data(:,4)'; % PHIE

rpt_data = load('./rpt_load_cuongpt.txt','-ascii');
input_rpt = rpt_data(:,1:2)';
target_rpt = rpt_data(:,4)';

rpt_mesh = load('./rpt_load_cuongpt_mesh.txt');
input_rpt_mesh = rpt_mesh(:,1:2)';
target_rpt_mesh = rpt_mesh(:,3)';


x_rpt = rpt_data(:,1);
y_rpt = rpt_data(:,2);
z_rpt = rpt_data(:,4);

[input_mm, input_ps] = mapminmax(input_well);
[target_mm, target_ps] = mapminmax(target_well);

% Create a two-column vector of points at which to evaluate the density.
x_grid = 3500:50:16000;
y_grid = 1.5:0.01:2.8;
[x_mesh,y_mesh] = meshgrid(x_grid,y_grid);
mesh_data = [x_mesh(:) y_mesh(:)];

%----------------- Generate initial weights and bias--------------------------
n  = [5];
k = [2;n';1];
net = feedforwardnet(n,'trainscg');
net.divideFcn = 'dividerand';  % Divide the data randomly (default)
net.divideParam.trainRatio = 0.7;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.15;
net.layers{1}.transferFcn='tansig';
net.layers{2}.transferFcn='tansig';

[net,tr] = train(net,input_well,target_well);

T = numel(k)-1;
W0 = cell(T,1);
b0 = cell(T,1);
wb0 = getwb(net);
[W0,b0] = wb_convert(wb0,k);

ObjectiveFunction = @costfunction;
w_gra_dir = 10;% gra_dir_costfunction
w_gra_amp = 100000; % gra_amp_costfunction
w_mat = 1000; % mat_costfunction
w_mat_rpt = 10;% mat  with rpt costfunction

fval0 = ObjectiveFunction(wb0);
fprintf('The best initial function value found was : %g\n', fval0);

% ---------------------- Simulated annealing -----------------------
options = optimoptions('simulannealbnd','PlotFcn',{@saplotbestf,@saplottemperature,@saplotf,@saplotstopping},'TemperatureFcn',{'temperatureboltz'});

% options.FunctionTolerance = 1e-5;
options.MaxTime = (24*60*60);
options.MaxStallIterations  = 10000;
ub = wb0 + 2;
lb = wb0 - 2;

% [wb_sa,fval_sa,exitFlag,output] = simulannealbnd(ObjectiveFunction,wb0,lb,ub,options);
% print('SA_resutls.png','-r300','-dpng')
% fprintf('The number of iterations was : %d\n', output.iterations);
% fprintf('The number of function evaluations was : %d\n', output.funccount);
% fprintf('The best SA function value found was : %g\n', fval_sa);

%---------------------- fminsearch --------------------------------
options = optimset('Display','iter','PlotFcns',@optimplotfval);
options.MaxIter = 30;
[wb_fmin,fval_fmin]  = fminsearch(ObjectiveFunction,wb0,options);
fprintf('The best FMIN function value found was : %g\n', fval_fmin);

%----------------- Apply optimal weights and bias for mesh data ---------------
% litho_predict_mesh = ksdensity_litho_func(well_data(:,1:3),mesh_data); % Estimating density 
% figure();hold on; gscatter(litho_predict_mesh(:,1),litho_predict_mesh(:,2),litho_predict_mesh(:,3),'kycbgr','*oooo',4,'filled');scatter(well_data(:,1), well_data(:,2),20, well_data(:,3),'filled','s');caxis([0 0.4]);hold off;
% 
% %----------------- Plot original data vs RPT ---------------------
% figure();hold on;
% scatter(well_data(:,1),well_data(:,2),30,well_data(:,4),'filled','o');
% plot_rpt('RPT_BII.1-10_points.xlsx')
% s = scatter(rpt_data(:,1),rpt_data(:,2),40,rpt_data(:,4),'filled','o');
% s.LineWidth = 0.6;
% s.MarkerEdgeColor = 'b';


net_opt0 = setwb(net,wb0);
net_out_mesh_0 = net(mesh_data')';
% net_opt_sa = setwb(net,wb_sa);
% net_out_mesh_sa = net_opt_sa(mesh_data')';

% net_opt_fmin = setwb(net,wb_fmin);
% net_out_mesh_fmin = net_opt_fmin(litho_predict_mesh(:,1:3)')';
% wb_compare = [wb0;wb_fmin;wb_sa];

figure()
subplot(1,2,1);
scatter(mesh_data(:,1),mesh_data(:,2),15,net_out_mesh_0,'filled');title('Initial weights and bias');colorbar;caxis([0 0.35]);overlay_rpt;
% figure();scatter(litho_predict_mesh(:,1),litho_predict_mesh(:,2),15,net_out_mesh_fmin,'filled');title('Fmin weights and bias');colorbar;caxis([0 0.4]);overlay_logs_rpt;
% subplot(1,2,2);
% scatter(mesh_data(:,1),mesh_data(:,2),15,net_out_mesh_sa,'filled');title('SA weights and bias');colorbar;caxis([0 0.35]);overlay_rpt;
% print('figure_compare_mesh_SA_FF.png','-r300','-dpng')


gra_amp_maps(wb_sa);
gra_dir_maps(wb_sa);

plotfmin(w_gra_dir,'fval_gradir.txt',w_gra_amp,'fval_graamp.txt',w_mat,'fval_mat.txt',w_mat_rpt,'fval_mat_rpt.txt');
print('figure_fmin.png','-r300','-dpng')

%---------------------------------- save results -----------------------%
save wb_sa.mat wb_sa
save wb0.mat wb0
dirname = datestr(datetime);
% dirname = strrep(dirname,':','_');dirname = strrep(dirname,' ','_');
% mkdir(['./SA_results/' '/' dirname '/']);
% fullpath = "./SA_results/" + dirname + "/";
% copyfile('mlp_sa_v2.m',fullpath);
% copyfile('fval_gradir.txt',fullpath);
% copyfile('fval_graamp.txt',fullpath);
% copyfile('fval_mat.txt',fullpath);
% copyfile('fval_mat_rpt.txt',fullpath);
% copyfile('figure_gra_dir_maps.png',fullpath);
% copyfile('figure_gra_amp_maps.png',fullpath);
% copyfile('figure_compare_mesh_SA_FF.png',fullpath);
% copyfile('figure_fmin.png',fullpath);
% copyfile('SA_resutls.png',fullpath);
% copyfile('wb_sa.mat',fullpath);
% copyfile('wb0.mat',fullpath);

function out = costfunction(x)
global w_gra_dir w_gra_amp w_mat w_mat_rpt
out =  w_gra_dir*gra_dir_costfunction(x) + w_gra_amp*gra_amp_costfunction(x) + w_mat*mat_costfunction(x) + w_mat_rpt*mat_rpt_costfunction(x);
end
