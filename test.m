%% Clear

clc; clf; clear; close all;

%% Pathes

addpath('funcs')    % Add path to the directory containing function files 
                    % required for the algorithm.
addpath('utils')    % Add path to the directory containing utility files 
                    % and helper functions.
addpath('results')  % Add path to the directory for saving and accessing 
                    % output results (figures, data, logs).

%% Load

X_true = double(imread('cameraman.tif'));
% X_true = double(imread('coins.png'));
%X_true = double(imread('rice.png'));

fig_win=figure;
imshow(X_true,[])
folder = 'results'; 
filename = 'real image.png';
title(filename);
fullFileName = fullfile(folder, filename);
saveas(fig_win, fullFileName);  

%% Normalization、Blur、Noise

[m,n] = size(X_true);
X_true_normalized=rescale(X_true,0,1);
A_op = fspecial('gaussian',9,4);
X_blurred = imfilter(X_true_normalized, A_op, 'symmetric', 'conv');
X_blurred_nosied = imnoise(X_blurred,'gaussian',0,1e-6);
B=X_blurred_nosied;

fig_win=figure;
imshow(X_blurred_nosied, [])
folder = 'results'; 
filename = 'blurred image.png';
title(filename);
fullFileName = fullfile(folder, filename);
saveas(fig_win, fullFileName);  

%% Options

options.max_outter_it = 350; 
options.A_op = 'symmetric_conv';
options.L=10;
options.alpha=1;
options.W_config='None';
options.x_true=X_true_normalized;
options.display_in_table=0;
options.display_in_figure=~options.display_in_table;

%% Compute

algos={ 
    'ISTA_CS';
    'ISTA_BT';
    'FISTA_CS';
    'FISTA_BT';
    'LBA'
        };

dim=size(algos);
X=cell(dim);
e=cell(dim);
t=cell(dim);
it=cell(dim);

for i=1:numel(algos)
    options.algo = algos{i};
    [X{i},e{i},t{i},it{i}] = main_LIP(A_op,B,options);
    fig_win=figure;
    imshow(X{i}, [])
    folder = 'results'; 
    filename = strcat([options.algo '_' options.W_config '.png']);
    title(strrep(filename,'_','\_'));
    fullFileName = fullfile(folder, filename);
    saveas(fig_win, fullFileName);  
end

%% Output&&Plot

if options.display_in_table==1
    fprintf('algo       | error      | time       | iteration  |\n');
    for i = 1:numel(algos)
        fprintf('%-10s | %-10f | %-10f | %-10d |\n', algos{i}, e{i}, t{i}, it{i});
    end
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