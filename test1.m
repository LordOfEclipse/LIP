%% Clear

clc; clf; clear; close all;

%% Pathes

addpath('funcs')    % Add path to the directory containing function files 
                    % required for the algorithm.
addpath('utils')    % Add path to the directory containing utility files 
                    % and helper functions.
addpath('results')  % Add path to the directory for saving and accessing 
                    % output results (figures, data, logs).

%% Generate、Noise

rng(randi(10000)); 

n=100;
theta=0.4;
A_op=sprand(n,n,theta);
A_op=A_op+10*speye(n,n);
b=sprand(n,1,theta);
b=rescale(b,0,1);
x_true=A_op\b;

b_nosied = imnoise(b,'gaussian',0,1e-6);

rel_e = norml2(A_op\b_nosied-x_true)/norml2(x_true);
fprintf('The relative error between the noisy solution and the true solution is: %f\n', rel_e);

%% Options

options.tol = 1e-2; 
options.max_outter_it = 50000; 
options.L=4*norml2(A_op'*A_op);
options.A_op = 'mult';
options.W_config='None';
options.x_true=x_true;
options.display_in_table=0;
options.display_in_figure=~options.display_in_table;

%options.B_config='1r';    %收敛很慢
%options.B_config='1c';    %收敛很慢
%options.B_config='2r';    %收敛很慢
%options.B_config='2c';    %收敛很慢
%options.B_config='2rs';   %收敛很慢
%options.B_config='2cs';   %不收敛

%% Compute

algos={ 
    'ISTA_CS';
    'ISTA_BT';
    'FISTA_CS';
    'FISTA_BT';
    %'BA_ISTA_CS';
    %'BA_FISTA_CS';
        };

dim=size(algos);
x=cell(dim);
e=cell(dim);
t=cell(dim);
it=cell(dim);

for i=1:numel(algos)
    options.algo = algos{i};
    [x{i},e{i},t{i},it{i}] = main_LIP(A_op,b_nosied,options);
end

%% Output&&Plot

if options.display_in_table==1
    fprintf('===================================================\n');
    fprintf('algo       | error      | time       | iteration  |\n');
    for i = 1:numel(algos)
        fprintf('%-10s | %-10f | %-10f | %-10d |\n', algos{i}, e{i}, t{i}, it{i});
    end
    fprintf('===================================================\n');
end

if options.display_in_figure==1

    % Legends
    legends = cellfun(@(x) strrep(x, '_', '\_'), algos, 'UniformOutput', false);
    
    % Create a new figure window
    fig_win = figure;
    
    % Create the first subplot
    subplot(1, 2, 1);
    
    for i = 1:numel(algos)
        semilogy(1:numel(e{i}), e{i}, 'LineWidth', 2);
        hold on;
    end
    
    % Set axis labels
    xlabel('Iterations');
    ylabel('Error');
    title('Convergence');
    
    % Add legends
    legend(legends, 'Location', 'best');
    
    % Add grid lines
    grid on;
    
    % Create the second subplot
    subplot(1, 2, 2);
    
    for i = 1:numel(algos)
        semilogy(t{i}, e{i}, 'LineWidth', 2);
        hold on;
    end
    
    % Set axis labels
    xlabel('Time');
    ylabel('Error');
    title('Convergence');
    
    % Add legends
    legend(legends, 'Location', 'best');
    
    % Add grid lines
    grid on;
    
    % Adjust figure size
    set(gcf, 'Position', [100, 100, 1600, 600]);
    
    % Save figure
    folder = 'results'; 
    filename = strcat(['Convergence' '.png']);
    fullFileName = fullfile(folder, filename);
    saveas(fig_win, fullFileName);   

end