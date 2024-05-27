GPU = 1;
parallel.gpu.enableCUDAForwardCompatibility(true)
ParaBasic = BasicParameters(5);



%Combine_HHData('Version','Benchmark','Type','Continuous');

%% Policy Intervention 1: Reduce increase of living costs
HousingCoefAdjust = [0.75 0.85 0.95 0.98];
ParaPol = struct();

for Coef = HousingCoefAdjust
    Version = strcat('HousingCoef_',num2str(Coef));
    %ParaPol.HousingCoef = ParaBasic.HousingCoef*Coef;
    
    %VF_final_discrete('GPU',GPU,'Version',Version,'LoadResultsVersion','Benchmark','ParaPol',ParaPol);
    %VF_final_continuous('GPU',GPU,'Version',Version);
    %Household_Simulations_Final('GPU',GPU,'Version',Version);
    Combine_HHData('Version',Version,'Type','Continuous');
end

%% Policy Intervention 2: Relocation subsidy
% DeltaW = 0.0:0.02:0.04;
% Version = 70;
% ParaPol = struct();
%
% 
% for Coef = DeltaW
%     Version = Version + 1;
%     ParaPol.DeltaW = Coef;
% 
%     Gridsizes = [10,30,30,10,5];
%     [VF0,Policy0,Grid0,Times0] = VF_final_discrete(GPU,Gridsizes,Version,ParaPol);
%     [HHDecision,HHPath] = Household_Simulations_Discrete(GPU,strcat('Results_discrete_',num2str(Version),'.mat'));
%     DATA = Combine_HHData(Version,'discrete');
%     Gridsizes = [20,50,50,15,5];
%     [VF0,Policy0,Grid0,Times0] = VF_final_discrete(GPU,Gridsizes,Version,[],Version);
%     [HHDecision,HHPath] = Household_Simulations_Discrete(GPU,strcat('Results_discrete_',num2str(Version+100),'.mat'));
%     DATA = Combine_HHData(Version+100,'discrete');
%     [VF2,Policy2,Grid2,SUMMARY] = VF_final_continuous(GPU,Version);
%     [HHDecision2,HHPath2] = Household_Simulations_Final(GPU,strcat('Results_continuous_',num2str(Version),'.mat'));
%     DATA2 = Combine_HHData(Version);
% end


%% Policy Intervention 3: Reduce prevention cost
PreventionCoefAdjust = [0.3 0.45 0.6 0.65 0.75];

ParaPol = struct();

for Coef = PreventionCoefAdjust
    Version = strcat('pFactor_',num2str(Coef));
%    ParaPol.pFactor = ParaBasic.pFactor*Coef;
%    VF_final_discrete('GPU',GPU,'Version',Version,'LoadResultsVersion','Benchmark','ParaPol',ParaPol);
%    VF_final_continuous('GPU',GPU,'Version',Version);
%    Household_Simulations_Final('GPU',GPU,'Version',Version);
    Combine_HHData('Version',Version,'Type','Continuous');
end


%% Policy Intervention 4: Minimum income
YminAdjust = 0.6:0.2:1.6;
ParaPol = struct();

for Coef = YminAdjust
    Version = strcat('YminAdjust_',num2str(Coef));
%    ParaPol.Ymin = ParaBasic.Y.Ymean(1)*Coef;
%    VF_final_discrete('GPU',GPU,'Version',Version,'LoadResultsVersion','Benchmark','ParaPol',ParaPol);
%    VF_final_continuous('GPU',GPU,'Version',Version);
%    Household_Simulations_Final('GPU',GPU,'Version',Version);
    Combine_HHData('Version',Version,'Type','Continuous');
end


%% Policy Intervention 5: Exposure dependent transfer
TransferCoef = [0.1 0.3 0.5 0.6 0.7 0.9];
TransferCoef = 0.8;
ParaPol = struct();
for Coef = TransferCoef
    Version = strcat('TransferCoef_',num2str(Coef));
    ParaPol.ETransfer = @(E) max(0.0, Coef*ParaBasic.Y.Ymean(1)*(E-0.5));
    VF_final_discrete('GPU',GPU,'Version',Version,'LoadResultsVersion','Benchmark','ParaPol',ParaPol);
    VF_final_continuous('GPU',GPU,'Version',Version);
    Household_Simulations_Final('GPU',GPU,'Version',Version);
    Combine_HHData('Version',Version,'Type','Continuous');
end



%% Policy Intervention 6: More convex living costs

GPU = 1;
parallel.gpu.enableCUDAForwardCompatibility(true)
HousingPowerNew = 2.4;%[2.2 2.5];
ParaPol = struct();

for Coef = HousingPowerNew
    Version = strcat('HousingPower_',num2str(Coef));
    ParaPol.HousingPower = Coef;
    
    VF_final_discrete('GPU',GPU,'Version',Version,'LoadResultsVersion','Benchmark','ParaPol',ParaPol);
    VF_final_continuous('GPU',GPU,'Version',Version);
    Household_Simulations_Final('GPU',GPU,'Version',Version);
    Combine_HHData('Version',Version,'Type','Continuous');
end

GPU = 2;
parallel.gpu.enableCUDAForwardCompatibility(true)

HousingPowerNew = [4.0,6.0];
ParaPol = struct();

for Coef = HousingPowerNew
    Version = strcat('HousingPower_',num2str(Coef));
    ParaPol.HousingPower = Coef;
    
    VF_final_discrete('GPU',GPU,'Version',Version,'LoadResultsVersion','Benchmark','ParaPol',ParaPol);
    VF_final_continuous('GPU',GPU,'Version',Version);
    Household_Simulations_Final('GPU',GPU,'Version',Version);
    Combine_HHData('Version',Version,'Type','Continuous');
end


%% 
HousingCoefAdjust = [0.75 0.85 0.95];
for Coef = HousingCoefAdjust
    Version = strcat('HousingCoef_',num2str(Coef));
    Household_Simulations_Final('GPU',GPU,'Version',Version);
    Combine_HHData('Version',Version,'Type','Continuous');
end

PreventionCoefAdjust = [0.45 0.55 0.75];
for Coef = PreventionCoefAdjust
    Version = strcat('pFactor_',num2str(Coef));
    Household_Simulations_Final('GPU',GPU,'Version',Version);
    Combine_HHData('Version',Version,'Type','Continuous');
end

YminAdjust = [0.6 1.2 1.6];
for Coef = YminAdjust
    Version = strcat('YminAdjust_',num2str(Coef));
    Household_Simulations_Final('GPU',GPU,'Version',Version);
    Combine_HHData('Version',Version,'Type','Continuous');
end

TransferCoef = [0.3 0.6 0.9];
for Coef = TransferCoef
    Version = strcat('TransferCoef_',num2str(Coef));
    Household_Simulations_Final('GPU',GPU,'Version',Version);
    Combine_HHData('Version',Version,'Type','Continuous');
end

