%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Partial Reinforcement Optimizer: An Evolutionary Optimization Algorithm %
% Running on Multiple Function IDs and Saving Results to Excel            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;
clc;

% -------------------- Settings -------------------------------- %
nPop = 30;                % Population Size
N = 10;                   % Number of Decision variables 
MaxFEs = N * 1000;       % Maximum number of function evaluations
NumofExper = 30;          % Number of experiments per function ID
LB = -100;                % Lower Bound
UB = 100;                 % Upper Bound

global initial_flag
initial_flag = 0;

% Excel setup
filename = 'PRO_Levi.xlsx';

% Loop over each function ID from 1 to 30 (excluding 2)
for Func_id = 1:30 
    % Objective function setup
    fhd = str2func('cec17_func');
    CostFunction = Func_id;
    LB_vec = LB .* ones(1, N);       
    UB_vec = UB .* ones(1, N);       

    % Initialize arrays for storing results of all experiments
    BestSolCostPRO = zeros(NumofExper, 1);
    BestSolCostPrev = zeros(NumofExper, 1);
    SumBestCostPRO_ = zeros(MaxFEs, 1);
    SumBestCostPrev_ = zeros(MaxFEs, 1);
    
    for ii = 1:NumofExper
        % Reset random seed and initial_flag for each experiment
        rand('state', sum(100 * clock));
        initial_flag = 0;

        % Create Initial Population
        Population = repmat(struct('Position', [], 'Cost', []), nPop, 1);
        for i = 1:nPop
            Population(i).Position = LB_vec + rand(1, N) .* (UB_vec - LB_vec);   
            Population(i).Cost = feval(fhd, Population(i).Position', CostFunction); 
        end  

        % Run PRO algorithm
        [BestCostPRO_, BestSolCostPRO(ii)] = Cauchy(N, MaxFEs, LB_vec, UB_vec, Population, nPop, CostFunction);
        SumBestCostPRO_ = SumBestCostPRO_ + BestCostPRO_(1:MaxFEs);

        % Run Prev algorithm
        [BestCostPrev_, BestSolCostPrev(ii)] = Prev(N, MaxFEs, LB_vec, UB_vec, Population, nPop, CostFunction);
        SumBestCostPrev_ = SumBestCostPrev_ + BestCostPrev_(1:MaxFEs);

        disp(['Function ID: ', num2str(Func_id), ' | Experiment ', num2str(ii), ...
              ' | New Best Cost: ', num2str(BestSolCostPRO(ii)), ...
              ' | Prev Best Cost: ', num2str(BestSolCostPrev(ii))]);
    end

    % Average results across experiments
    AveBestCostPRO_ = SumBestCostPRO_ ./ NumofExper;
    AveBestCostPrev_ = SumBestCostPrev_ ./ NumofExper;

    % Statistics for comparison
    MeanPRO = mean(BestSolCostPRO);
    SDPRO = std(BestSolCostPRO);
    MeanPrev = mean(BestSolCostPrev);
    SDPrev = std(BestSolCostPrev);
    
    % Save results for current Func_id to Excel
    sheet_name = ['Func_' num2str(Func_id)];
    T = table((1:NumofExper)', BestSolCostPRO, BestSolCostPrev, 'VariableNames', ...
              {'Experiment', 'BestSolCostPRO', 'BestSolCostPrev'});
    writetable(T, filename, 'Sheet', sheet_name, 'Range', 'A1');
    
    % Add summary statistics to Excel
    summary_data = table({'Mean'; 'StdDev'}, [MeanPRO; SDPRO], [MeanPrev; SDPrev], ...
                         'VariableNames', {'Statistic', 'PRO', 'Prev'});
    writetable(summary_data, filename, 'Sheet', sheet_name, 'Range', 'E1');

    % Plot and save convergence data
    fig = figure;
    semilogy(AveBestCostPRO_, 'r-', 'LineWidth', 2);
    hold on;
    semilogy(AveBestCostPrev_, 'b-', 'LineWidth', 2);
    grid on;
    xlabel('Function Evaluations');
    ylabel(['F(x) = Func ', num2str(Func_id)]);
    title(['Comparison of PRO and Prev for Function ID ', num2str(Func_id)]);
    legend('PRO', 'Prev (Dynamic RR)');
    hold off;
    saveas(fig, ['Convergence_Func_' num2str(Func_id) '.png']);
    close(fig);  % Close the figure after saving
end
