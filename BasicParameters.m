function Para = BasicParameters(nY)

%% Utility function
% Weight of wealth relative to consumption in utility
Para.theta = 0.75;           % NOT IN DATA
% CRRA-parameter in utility function for wealth
Para.beta = 1.1;            % NOT IN DATA
% CRRA-parameter in utility function for consumption
Para.gamma = 1.1;           % NOT IN DATA
% Scale factor for absolute value of consumption and wealth in utility function
Para.scalefactor = 1.0;       % NOT IN DATA


%% Other Parameters

% depreciation rate of durable consumption goods/wealth over 2 years (10% per year)
Para.delta = 1-(1-0.025)^2;              %Try to find in data!
% interest rate over 2 years (4% per year)
Para.r = (1+0.04)^2 - 1;%0.04;%         %Try to find in data!
Para.rP = (1+0.006871891)^2 - 1;        %Mean interest rate on savings
Para.rM = (1+0.05495146)^2 - 1;         %Mean borrowing interest rate in data

% Share of income lost in next period due to disaster
Para.Deltay =  0.08336988;              %CHECK
% Share of wealth lost due to relocation of settlement
Para.DeltaW = 0.1;                     % NOT IN DATA
% Share of wealth remaining when liquidating it into financial savings
Para.kappaZ = 0.9;                      % NOT IN DATA

%% Income Process (FROM DATA)
% Years of Education
Para.lowH0 = 1.0;
Para.highH0 = 5.0;
Para.nH0 = 5;

% Rate of return for one year of education
Para.Y.Ymean = exp([0.0 0.2717758 0.9821299 1.2781891 1.8577960]);  %CHECK
% Time correlation of income (persistence)
Para.Y.rho = [0.2111935 0.2141772 0.1778366 0.1548218 0.5424555];   %CHECK
% Standard deviation of log-income
Para.Y.sigma = [1.0208932 1.1252863 0.8547407 0.7523482 0.9548220]; %CHECK
% Parameter for Tauchen-Approximation
Para.Y.lambda = 1.0;
Para.Ymin = 0.0;

%% Lower and Upper Bounds
% Minimum consumption level for survival
if nargin < 1
    nY = 5;
end

Para.YBase = tauchen(nY,Para.Y.rho(1),Para.Y.sigma(1),Para.Y.lambda);
Para.Ylow = Para.YBase(1)*(1-Para.Deltay);    % IN DATA (4.4% percentile in data of lowest education group)
Para.Yhigh = Para.YBase(end);         % IN DATA (95.3% percentile in data of lowest education group)

% Minimum consumption level for survival
Para.lowC = 0.0;%0.1578688;
% Minimum wealth level for survival
Para.lowW = 0.0;%0.1017095;
% Upper bound for wealth level
Para.highW = [20.0,20.0,30.0,30.0,50.0];          % IN DATA

% Lower bound for exposure level
Para.lowE = 0.01;          
% Upper bound for exposure level          
Para.highE = 1.0;           
% Lower bound for financial savings
Para.lowA = -33.84798;              
% Upper bound for financial savings
Para.highA = [15.0,15.0,15.0,15.0,20.0];   %2.988863 1.692669 2.433048 3.741826 5.218818

Para.exponentA = 2.0;
Para.exponentW = 2.0;

%% Cost functions
Para.HousingCoef = 1.659999;%1.075065;
Para.HousingMax = 1.378308;
Para.HousingMin = 0.2620703; %(Para.Ylow - Para.lowC - Para.lowW)*0.99;

if Para.HousingMin > (Para.Ylow - Para.lowC - Para.lowW)*0.99
    Para.HousingMin = (Para.Ylow - Para.lowC - Para.lowW)*0.99;
    Para.HousingAdjust = log(Para.HousingMax/Para.HousingMin)/Para.HousingCoef + 1;
else
    Para.HousingAdjust = 1;
end

Para.HousingPower = 2;
Para.pEfunction = @(E) Para.HousingMin*exp(Para.HousingCoef*(1-E)^Para.HousingPower);
Para.ETransfer = @(E) 0.0;

% Prevention cost function
Para.pFactor = (1-Para.delta);%1.0;
  

%% Household characteristics

% Awareness factor (in [0,1])
Para.lowa = 0.5;  
Para.higha = 0.9;
Para.na = 3;

% Time discount rate over 1 period
% Discount rate has to be higher than the interest rate to obtain a useful solution (otherwise there would be infinite aggregating of savings
Para.lowrho = Para.rM*1.4;                     
Para.highrho = Para.rM*2.3;
Para.nrho = 3;

% Education impact on prevention
Para.lowphiP = 0.1;
Para.highphiP = 0.6;
Para.nphiP = 3;

end
