GPU = 1;
parallel.gpu.enableCUDAForwardCompatibility(true)
Version='Benchmark';


Gridsizes = [15,40,40,20,5];

VF_final_discrete('GPU',GPU,'Gridsizes',Gridsizes,'Version',Version);
Household_Simulations_Discrete('GPU',GPU,'Version',Version);
Combine_HHData('Version',Version,'Type','Discrete');

% Gridsizes = [15,40,40,15,5];
% VF_final_discrete('GPU',GPU,'Gridsizes',Gridsizes,'Version','Test2','LoadResultsVersion','Test');

VF_final_continuous('GPU',GPU,'Version',Version);
Household_Simulations_Final('GPU',GPU,'Version',Version);
Combine_HHData('Version',Version,'Type','Continuous');



%% Test

ParaTest = struct();
ParaTest.theta = 0.7; 
ParaTest.beta = 1.05; 
ParaTest.gamma = 1.05;

ParaTest.lowa = 0.4;  
ParaTest.higha = 0.8;
ParaTest.na = 3;


Version='Test2';
Gridsizes = [15,40,40,20,5];

VF_final_discrete('GPU',2,'Gridsizes',Gridsizes,'Version',Version,'ParaPol',ParaTest);
Household_Simulations_Discrete('GPU',2,'Version',Version);
Combine_HHData('Version',Version,'Type','Discrete');

% Gridsizes = [15,40,40,15,5];
% VF_final_discrete('GPU',GPU,'Gridsizes',Gridsizes,'Version','Test2','LoadResultsVersion','Test');

VF_final_continuous('GPU',2,'Version',Version);
Household_Simulations_Final('GPU',2,'Version',Version);
Combine_HHData('Version',Version,'Type','Continuous');