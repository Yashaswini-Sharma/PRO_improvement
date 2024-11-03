

function [BestCosts,BestSolCost]= Cauchy(N,MaxFEs,LB,UB,population,pop_size,ObjFunc_ID)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Partial Reinforcement Optimizer: An Evolutionary Optimization Algorithm %
% version 1.0                                                             %
% Authiors:                                                               %
% Ahmad Taheri **, Keyvan RahimiZadeh, Amin Beheshti, Jan Baumbach,       %
% Ravipudi Venkata Rao, Seyedali Mirjalili, Amir H Gandomi                %
%                                                                         %
% ** E-mail:                                                              %
%          Ahmad.taheri@uni-hamburg.de                                    %
%          Ahmad.thr@gmail.com                                            %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialization 
rand('state',sum(100*clock));
% --- Problem Definition ---
function L = levyFlight(beta, nVar)
    % Levy flight generator using Mantegna's algorithm
    % beta is the parameter, typically in the range [1.5, 2]
    % nVar is the number of variables
    
    % Constants
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) / ...
             (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta);
    
    % Generate Levy step
    u = randn(1, nVar) * sigma;
    v = randn(1, nVar);
    L = u ./ abs(v).^(1 / beta);
end
 

                                  
%Obj_Func = @ YourObjFunc;         %  Objective Function
fhd = str2func('cec17_func');
nVar = N;                          %  Number of Decision Variables
LB =  LB .* ones(1,nVar);          %  Variables Lower Bound
UB =  UB .* ones(1,nVar);          %  Variables Upper Bound

% --- PRO Parameters ---
RR = 0.9;                          %  Reinforcement Rate (RR)
%MaxFEs = MaxFEs;                  %  Maximum Number of Function Evaluations
nPop = pop_size;                   %  Population Size
FEs = 0;                           %  Function Evaluations counter

% --- Empty Structure for Individuals ---
empty_individual.Behaviors=[];
empty_individual.response=[];
empty_individual.Schedule=[];
%empty_individual.Fy=[];
%empty_individual.Gx=[];

% --- Initialize Population Array ---
pop = repmat(empty_individual, nPop, 1);
% --- Initialize Best Solution ---
BestSol.response = inf;
% --- Initialize Population --- 
for i=1:nPop
    pop(i).Behaviors = population(i).Position;
    pop(i).Schedule = unifrnd(0.9,1,1,N);
    pop(i).response =  population(i).Cost; %feval('cec14_func',pop(i).Behaviors',CostFunction)  - (CostFunction*100);
end
% --- Sort pop ---
[~,SorteIndx] = sort([pop.response]);
pop = pop(SorteIndx);
% --- Set the Best Solution ---
BestSol = pop(1);

% --- Initialize Best Cost Record ---
BestCosts = zeros(MaxFEs,1);
BestCosts(1) = BestSol.response;
[~, sortedIndx] = sort([pop.response]);

ResetZero = zeros(1,N);

%% --- PRO Main Loop ---
 %% --- PRO Main Loop ---
while FEs < MaxFEs 
   
    for i = 1:nPop  % For all Learners      
       tempBehav = pop(i);     
     
       k = nPop;
       if i < nPop
         k = sortedIndx(randi([i+1 nPop]));
       end    
              
       %% --- Determine Behaviors of the ith learner based on Scheduler. -----------  
       Tau = (FEs / MaxFEs);
       RR = 0.9 - 0.8 * Tau;
       Selection_rate = exp(-(1 - Tau));
       [~, Candid_Behavs] = sort(pop(i).Schedule(1:N), 'descend');
       Landa = ceil(N * rand * Selection_rate);
       Landa = min(Landa, length(Candid_Behavs));
       Selected_behaviors = Candid_Behavs(1:Landa);
                
       %% --- Stimulate the selected Behaviors of the ith learner to get response.---  
       if rand <  0.5
          Stimulation = ResetZero;
          Stimulation(Selected_behaviors) = (BestSol.Behaviors(Selected_behaviors) - pop(i).Behaviors(Selected_behaviors));    
       else
          Stimulation = ResetZero;
          Stimulation(Selected_behaviors) = (pop(i).Behaviors(Selected_behaviors) - pop(k).Behaviors(Selected_behaviors));            
       end
       
       % --- Calculate Stimulation Factor with Lévy flight -----
       SF = Tau + rand * (mean((pop(i).Schedule(Selected_behaviors)) / max(abs(pop(i).Schedule))));
       
       % Lévy step addition
       beta = 1.5; % Parameter for Lévy flight, commonly set around 1.5
       LevyStep = levyFlight(beta, Landa); % Generate Lévy step
       
       % Update behaviors with Lévy step
       tempBehav.Behaviors(Selected_behaviors) = pop(i).Behaviors(Selected_behaviors) + SF .* Stimulation(Selected_behaviors) + 0.01 * LevyStep;
       
       % ------------ Bound constraints control ------------------- 
       [~, underLB] = find(tempBehav.Behaviors < LB);
       [~, uperUB] = find(tempBehav.Behaviors > UB);
       if ~isempty(underLB)
         tempBehav.Behaviors(underLB) =  LB(underLB) + rand(1, size(underLB, 2)) .* ((UB(underLB) - LB(underLB)) / 1); 
       end
       if ~isempty(uperUB)
         tempBehav.Behaviors(uperUB) =  LB(uperUB) + rand(1, size(uperUB, 2)) .* ((UB(uperUB) - LB(uperUB)) / 1); 
       end
       
       % ------ Evaluate the ith learner Response -------------------
       tempBehav.response = feval(fhd, tempBehav.Behaviors', ObjFunc_ID); % CEC2017
     
       FEs = FEs + 1;
       
       % ----- Apply Positive or Negative Reinforcement according to the response.
       if tempBehav.response < pop(i).response
            % Positive Reinforcement 
            tempBehav.Schedule(Selected_behaviors) = pop(i).Schedule(Selected_behaviors) + pop(i).Schedule(Selected_behaviors) * (RR / 2);           
            
            % accept new Solution
            pop(i) = tempBehav;
            
            % Update the best Solution
            if pop(i).response < BestSol.response
                BestSol = pop(i);  
            end
       else
            % Negative Reinforcement
            pop(i).Schedule(Selected_behaviors) = pop(i).Schedule(Selected_behaviors) - pop(i).Schedule(Selected_behaviors) * (RR);
       end
       
       % Store Record for Current Iteration
       BestCosts(FEs) = BestSol.response; 
       
       %% ------- Rescheduling --------------------------------------------------    
       if std(pop(i).Schedule(1:N)) == 0
           pop(i).Schedule = unifrnd(0.9, 1, 1, N);
           pop(i).Behaviors = LB + rand(1, N) .* (UB - LB);
           pop(i).response = feval(fhd, pop(i).Behaviors', ObjFunc_ID); % CEC2017
           disp(['-------------------------------- The Learner ' num2str(i) ' is Rescheduled ']);
       end

   end % End for nPop
   
   %% Sort pop
   [~, SorteIndx] = sort([pop.response]);
   pop = pop(SorteIndx);
      
   % --- Show Iteration Information ---
   disp(['Iteration ' num2str(FEs) ': Best Cost = ' num2str(BestCosts(FEs))]);
end % End While
BestSolCost = BestSol.response;
end % Function Cauchy
