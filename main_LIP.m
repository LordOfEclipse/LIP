function [x,e,t,it] = main_LIP(A_op,b,options)
% *************************************************************************
% * This function aims to solve linear inverse problem of the form: 
%
%           min {J(x) = F(x) + R(x)},
%            x
%   
%   where   F(x) = ||A*x-b||_2^2        is the data-fidelity term,
%   and     R(x) = Lambda * r(x)        is the regularization term.
%
%           r(x) = ||x||_1
%                = ||Wx||_1 %ToDo
%                = Phi(x)   %ToDo
% *************************************************************************
% * References:
%   [1] J. M. Bioucas-Dias and M. A. T. Figueiredo, "A New TwIST: Two-Step
%       Iterative Shrinkage/Thresholding Algorithms for Image Restoration,"
%       IEEE Transactions on Image Processing 16, 2992-3004 (2007).
%   [2] A. Beck and M. Teboulle, "A Fast Iterative Shrinkage-Thresholding 
%       Algorithm for Linear Inverse Problems," SIAM Journal on Imaging 
%       Sciences 2, 183-202 (2009).
%
% *************************************************************************
% * Author:     Hu Dongnan.
% * Date:       2024/08/14.
% *************************************************************************
%
%   ===== Required inputs =================================================
%
%   - A_op :  2D float array [Required]          %(1D / 3D to be completed)
%            The structure contains operator information, and the program 
%            generates function handles accordingly.
%
%      options.A_op can be one of the following options
%         -> 'symmetric_conv' : then A_op is a convolution kernel
%                               while the boundary condition is
%                               symmetric.
%         -> 'mult' : A_op is a matrix that directly operate on x.
%
%   - b    : 1D / 2D float array  [Required]        %(3D to be examined)
%            Observation vector/matrix.
%
%   ===== Optional inputs =================================================
%
%   - options : struct
%               Various options for algorithm tuning and behavior.
%
%         -> dim :  int array [default: size(b)] 
%                   The dimension of the Observation vector b and
%                   Solution vector x.
%   
%         -> display :  boolean [default: false] 
%                       Determines whether the programme will output.
%   
%         -> display_in_figure : boolean  [default: false] 
%                                Display error in a figure window.
%   
%         -> display_in_table :  boolean  [default: true] 
%                                Display error in a table.
%   
%         -> A_op :  string [Required] 
%                    The operator A for the algorithm.
%                    ->> 'symmetric_conv' : A_op is a convolution kernel
%                                           while the boundary condition is
%                                           symmetric.
%                    ->> 'mult' : A_op is a matrix that directly operate on
%                                 x.
%   
%         -> algo :  string [default: 'FISTA_CS'] 
%                    The algorithm to be used. Select from the following 
%                    list:
%              ->> 'ISTA_CS' : Iterative Shrinkage-Thresholding Algorithm 
%                              with constant stepsize
%              ->> 'ISTA_BT' : Iterative Shrinkage-Thresholding Algorithm 
%                              with backtracking
%              ->> 'FISTA_CS' : Fast Iterative Shrinkage-Thresholding 
%                               Algorithm with constant stepsize
%              ->> 'FISTA_BT' : Fast Iterative Shrinkage-Thresholding 
%                               Algorithm with backtracking
%              ->> 'TWISTA' : Two-Step Inexact Proximal Gradient Algorithm
%                               %(ToDo).
%              ->> 'MTWISTA' : Multigrid version of TWISTA
%                               %(ToDo).
%              ->> 'IHTA_CS' : Iterative Hard Thresholding Algorithm
%                               %(ToDo).
%              ->> 'BIA' : Bregmann Iterative Algorithm
%                               %(ToDo).
%              ->> 'LBA' : Linearized Bregmann Algorithm
%                               %(ToBeDetermined).
%              ->> 'SBA' : Spilt Bregmann Algorithm
%                               %(ToDo).
%              ->> 'BA_ISTA_CS' : BA Iterative Shrinkage-Thresholding 
%                                 Algorithm with constant stepsize
%                               %(ToBeDetermined).
%   
%         -> B_config :  [Required] 
%                        Configuration for B to generate for replacing AT. 
%                        When using the 'BA_ISTA_CS' algorithm, the B_config 
%                        specifies how to compute the B operator which is 
%                        used in place of the traditional adjoint operator 
%                        AT in the algorithm. The B operator is typically 
%                        designed to promote certain properties in the 
%                        solution, such as sparsity.
%
%               The B_config field is mandatory when the chosen algorithm 
%               is 'BA_ISTA_CS'. It dictates the construction of the B 
%               operator based on the norms of the rows or columns of the 
%               operator A. Here are the different configurations available:
%
%               ->> '1r' :  Each element of B is the inverse of the L1 norm 
%                           of the corresponding row of A.
%               ->> '1c' :  Each element of B is the inverse of the L1 norm 
%                           of the corresponding column of A.
%               ->> '2r' :  Each element of B is the inverse of the square 
%                           of the L2 norm of the corresponding row of A.
%               ->> '2c' :  Each element of B is the inverse of the square 
%                           of the L2 norm of the corresponding column of A.
%               ->> '2rs' : Each element of B is computed using a custom 
%                           norm that accounts for the number of non-zero
%                           elements (S) in the row of A, and the L2 norm 
%                           of the row.
%               ->> '2cs' : Similar to '2rs', but applied to the columns 
%                           of A.
%
%               If the 'B_config' field is not present in the options 
%               structure, an error is raised indicating that B_config is 
%               required.
%   
%         -> max_outter_it : [default: 1000] 
%                            Maximum number of outer iterations.
%   
%         -> max_inner_it :  [default: 10] 
%                            Maximum number of inner iterations.
%   
%         -> Lambda :    [default: 2e-5] 
%                        Regularization parameter.
%   
%         -> L : [default: calculated based on A_op] 
%                Lipschitz constant for certain algorithms.
%   
%         -> t : Step size parameter for certain algorithms.
%   
%         -> eta : parameter for backtracking.
%   
%         -> alpha : parameter for LBA algorithms.
%
%         -> tol : Tolerance for the algorithm to determine convergence.
%   
%         -> error_upper_bound : Maximum error upper bound during 
%                                algorithm execution.
%
%         -> time_upper_bound : Maximum time upper bound during 
%                                algorithm execution.
%   
%         -> x_true_known :  [default: 0] 
%                            Indicates if the true solution x_true is 
%                            known.
%   
%         -> W_config : Configuration for the transform matrix W.
%                       This option specifies the type of transformation 
%                       matrix W to be used in the algorithm.
%                       The transformation matrix W can influence the 
%                       convergence and performance of the algorithm.
%
%                       ->> 'None' : Indicates no transformation will be 
%                                    applied.
%                       ->> 'Orthogonal' : [ToDo] An orthogonal transformation, 
%                                          which could potentially speed up 
%                                          convergence.
%                       ->> 'Fourier' : [ToDo] A Fourier transform-based 
%                                       matrix, useful for frequency domain 
%                                       analysis.
%                       ->> 'Wavelet' : [ToDo] A wavelet transform-based 
%                                       matrix, suitable for sparse 
%                                       representation of certain signals.
%
%                       If an unrecognized configuration is specified, an 
%                       error will be raised indicating that the transform 
%                       does not exist.
%   
%         -> stop_criterion : Criterion to determine when to stop the 
%                             algorithm.
%                             This option specifies the condition under 
%                             which the optimization algorithm will terminate.
%                             Different criteria can be used to balance 
%                             between the accuracy of the solution and the 
%                             computational efficiency. The following stop 
%                             criteria are available:
%
%                ->> 'change_rel_norm' : Stop if the relative change in the 
%                                        norm of the solution falls below a 
%                                        threshold.
%                ->> 'rel_norm' : Stop when the relative norm of the solution 
%                                        reaches a certain tolerance.
%                ->> 'rel_norm_from_truth' : Similar to 'rel_norm', but 
%                                            compared to the norm of the 
%                                            true solution if known.
%                ->> 'change_abs_norm' : Stop if the absolute change in the 
%                                        norm of the solution falls below a 
%                                        threshold.
%                ->> 'abs_norm' : Stop when the absolute norm of the solution 
%                                 reaches a certain tolerance.
%                ->> 'abs_norm_from_truth' : Similar to 'abs_norm', but 
%                                            compared to the norm of the 
%                                            true solution if known.
%                ->> 'obj_J' : Stop when the objective function J reaches a 
%                              certain value or tolerance.
%                ->> 'obj_F' : Stop based on the value or tolerance of the 
%                              data fidelity term F(x).
%                ->> 'obj_R' : Stop based on the value or tolerance of the 
%                              regularization term R(x).
%                ->> 'change_obj_J' : Stop if the relative change in the 
%                                     objective function J falls below a 
%                                     threshold.
%                ->> 'change_obj_F' : Stop if the relative change in the 
%                                     data fidelity term F(x) falls below a 
%                                     threshold.
%                ->> 'change_obj_R' : Stop if the relative change in the 
%                                     regularization term R(x) falls below 
%                                     a threshold.
%                ->> 'max_outter_it' : Stop after reaching the maximum 
%                                      number of outer iterations specified 
%                                      by 'max_outter_it'.
%
%               If the 'stop_criterion' field is not provided in the options, 
%               the algorithm will default to using 'rel_norm_from_truth' 
%               as the stopping criterion, which requires the knowledge of 
%               the true solution.
%
%   ===== Outputs =========================================================
%
%   - x : array
%         Solution vector/matrix.
%
%   - e : array
%         Error history during optimization.
%
%   - t : array
%         Time history during optimization.
%
% *************************************************************************

%% Pathes

addpath('funcs')    % Add path to the directory containing function files 
                    % required for the algorithm.
addpath('utils')    % Add path to the directory containing utility files 
                    % and helper functions.
addpath('results')  % Add path to the directory for saving and accessing 
                    % output results (figures, data, logs).

%% Options

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin < 2
    error('At least enter A_op and b.');
end
options.dim = size(b); 
if nargin < 3
    options = [];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isfield(options,'display')
    options.display = false; 
end
if ~isfield(options,'display_in_figure')
    options.display_in_figure = false; 
end
if ~isfield(options,'display_in_table')
    options.display_in_table = true; 
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isfield(options,'A_op')
    error("Operator didn't exist.");
end
switch options.A_op
    case 'symmetric_conv'
        A = @(x) imfilter(x, A_op, 'symmetric', 'conv');
        func_hand.A=@A;
        AT = @(x) imfilter(x, A_op', 'symmetric', 'conv');
        func_hand.AT=@AT;
    case 'mult'
        A = @(x) reshape(A_op*x(:),size(x));
        func_hand.A=@A;
        AT = @(x) reshape(A_op'*x(:),size(x));
        func_hand.AT=@AT;
        ATA_op=A_op'*A_op;
        ATA = @(x) reshape(ATA_op*x(:),size(x));
        func_hand.ATA=@ATA;
        ATb=A_op'*b;
        func_hand.ATb=ATb;
    otherwise
        error('Wrong operator.');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(options.algo, 'BA_ISTA_CS')||strcmp(options.algo, 'BA_FISTA_CS')
    if ~strcmp(options.A_op, 'mult')
        error('Wrong A_op.')
    end
    if ~isfield(options,'B_config')
        error("B_config didn't exist.")
    end
    [nr_A,nc_A]=size(A_op);
    switch options.B_config
        case '1r'
            B_op=zeros(1,nr_A);
            for i=1:nr_A
                B_op(i)=1/norml1(A_op(i,:));
            end
        case '1c'
            B_op=zeros(1,nc_A);
            for i=1:nc_A
                B_op(i)=1/norml1(A_op(:,i));
            end
        case '2r'
            B_op=zeros(1,nr_A);
            for i=1:nr_A
                B_op(i)=1/norml2(A_op(i,:))^2;
            end
        case '2c'
            B_op=zeros(1,nc_A);
            for i=1:nc_A
                B_op(i)=1/norml2(A_op(:,i))^2;
            end
        case '2rs'
            B_op=zeros(1,nr_A);
            S=zeros(1,nr_A);
            for i=1:nr_A
                S(i)=nnz(A_op(i,:));
            end
            for i=1:nr_A
                B_op(i)=1/normS(A_op(i,:),S)^2;
            end
        case '2cs'
            B_op=zeros(1,nc_A);
            S=zeros(1,nc_A);
            for i=1:nc_A
                S(i)=nnz(A_op(:,i));
            end
            for i=1:nc_A
                B_op(i)=1/normS(A_op(:,i),S)^2;
            end
    end
    A = @(x) reshape(A_op*x(:),size(x));
    func_hand.A=@A;
    AT = @(x) reshape(A_op'*x(:),size(x));
    func_hand.AT=@AT;
    ATBA_op=A_op'*B_op'.*A_op;
    ATBA = @(x) reshape(ATBA_op*x(:),size(x));
    func_hand.ATBA=@ATBA;
    ATBb=A_op'*B_op'.*b;
    func_hand.ATBb=ATBb;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isfield(options,'max_outter_it')
    options.max_outter_it = 1000; 
end
if ~isfield(options,'max_inner_it')
    options.max_inner_it = 10; 
end
if ~isfield(options,'Lambda')
    options.Lambda = 2e-5; 
end
if ~isfield(options,'L')
    if ~strcmp(options.A_op,'mult')
        options.L = 1; 
    else
        options.L=2*norml2(A_op'*A_op);
    end
end
if ~isfield(options,'t')
    options.t = 1; 
end
if ~isfield(options,'eta')
    options.eta = 1.1; 
end
if ~isfield(options,'alpha')
    options.alpha = 1; 
end
if ~isfield(options,'tol')
    options.tol = 1e-5; 
end
if ~isfield(options,'error_upper_bound')
    options.error_upper_bound = 1e2; 
end
if ~isfield(options,'time_upper_bound')
    options.time_upper_bound = 1e1; 
end
if isfield(options,'x_true')
    options.x_true_known = 1; 
else 
    options.x_true_known = 0; 
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if isfield(options,'W_config')
    switch options.W_config
        case 'None'
        %case 'Orthogonal'  %ToDo
        %case 'Fourier'     %ToDo
        %case 'Wavelet'     %ToDo
        otherwise
            error("Transform didn't exist.");
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isfield(options,'stop_criterion')
    options.stop_criterion='rel_norm_from_truth';
end
switch options.stop_criterion
    case 'change_rel_norm'
    case 'rel_norm'
    case 'rel_norm_from_truth'
        if ~options.x_true_known
            error("x_ture didn't know.");
        end
    case 'change_abs_norm'
    case 'abs_norm'
    case 'abs_norm_from_truth'
        if ~options.x_true_known
            error("x_ture didn't know.");
        end
    case 'obj_J'
    case 'obj_F'
    case 'obj_R'
    case 'change_obj_J'
    case 'change_obj_F'
    case 'change_obj_R'
    case 'max_outter_it'
    otherwise
        error("stop_criterion didn't exist.");
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isfield(options,'algo')
    options.algo = 'FISTA_CS'; 
end
switch options.algo
    case 'ISTA_CS'
        ALGO = @ISTA_CS;
    case 'ISTA_BT'
        ALGO = @ISTA_BT;
    case 'FISTA_CS'
        ALGO = @FISTA_CS;
    case 'FISTA_BT'
        ALGO = @FISTA_BT;
    case 'TWISTA'
        ALGO = @TWIST;      %ToDo
    case 'MTWISTA'
        ALGO = @MTWIST;     %ToDo
    case 'IHTA_CS'
        ALGO = @IHTA_CS;    %ToDo
    case 'BIA'
        ALGO = @BIA;        %ToDo
    case 'LBA'
        ALGO = @LBA;        %ToBeDetermined
    case 'SBA'
        ALGO = @SBA;        %ToDo
    case 'BA_ISTA_CS'
        ALGO = @BA_ISTA_CS;
    case 'BA_FISTA_CS'
        ALGO = @BA_FISTA_CS;
    otherwise
        error('Unknown algorithm specified.');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Auxilary Functions

r = @norml1;
proxr = @proxl1;

% calculate the data-fidelity term F(x)
function val = F(x)
    val = normArr(A(x)-b)^2;
end
func_hand.F=@F;

% calculate the regularization term R(x)
function val = R(x)
    val = options.Lambda*r(x);
end
func_hand.R=@R;

% calculate the objective function J(x)
function val = J(x)
    val = F(x) + R(x);
end
func_hand.J=@J;

% calculate the quadratic approximation function Q(x,y,L)
function val = Q(x,y,L)
    val = F(y) + dotArr(x-y,dF(y)) + L/2*normArr(x-y)^2 + R(y);
end
func_hand.Q=@Q;

% calculate the proximity operator for function R(x)
function val = proxR(x,gamma)
    val = proxr(x,gamma*options.Lambda);
end
func_hand.proxR=@proxR;

% calculate the gradiet for F(x)
function val = dF(x)
    if ~strcmp(options.algo, 'BA_ISTA_CS')&&~strcmp(options.algo, 'BA_FISTA_CS')
        switch options.A_op
            case 'symmetric_conv'
                val = AT(A(x) - b);
            case 'mult'
                val = ATA(x) - ATb;
        end
    else
        val = ATBA(x) - ATBb;
    end
end
func_hand.dF=@dF;

%% Main

[x,e,t,it] = ALGO(b,options,func_hand);

%% Display

if options.display&&options.display_in_figure

    % Create a new figure window
    figure('Name', [options.algo ' Convergence'], 'NumberTitle', 'off', 'Color', 'w');
    
    % Plot the error curve
    subplot(2,1,1);
    p1 = semilogy(1:numel(e), e, 'LineWidth', 2, 'Color', 'b');
    hold on;
    
    % Set axis labels
    xlabel('Iterations', 'FontSize', 14);
    ylabel('Error', 'FontSize', 14);
    title('Convergence', 'FontSize', 16);

    % Add grid lines
    grid on;
    set(gca, 'GridLineStyle', ':', 'GridColor', 'k', 'GridAlpha', 0.6);
    
    % Plot the error curve against time
    subplot(2,1,2);
    p2 = semilogy(t, e, 'LineWidth', 2, 'Color', 'r');
    
    % Set axis labels
    xlabel('Time', 'FontSize', 14);
    ylabel('Error', 'FontSize', 14);
    title('Convergence', 'FontSize', 16);

    % Add grid lines
    grid on;
    set(gca, 'GridLineStyle', ':', 'GridColor', 'k', 'GridAlpha', 0.6);
    
    % Add legend
    legend([p1, p2], {'Error', 'Time'}, 'Location', 'best', 'FontSize', 12);
    
    % Adjust figure size
    set(gcf, 'Position', [100, 100, 800, 600]);

elseif options.display&&options.display_in_table
    fprintf('===================================================\n');
    fprintf('algo       | error      | time       | iteration  |\n');
    fprintf('%-10s | %-10f | %-10f | %-10d |\n', options.algo, e, t, it);
    fprintf('===================================================\n');
end

end