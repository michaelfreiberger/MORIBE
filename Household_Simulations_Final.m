function [HHDecision,HHEndowment] = Household_Simulations_Final(varargin)
 
% DefaultValues
GPU = 1;
Version = 'Test';
LoadResultsVersion = '';
SaveFolder = "D:\Users\mfreiber\DisasterRiskModel\Matlab-Simulations_V2.1";

for jj = 1:2:nargin
    if strcmp('GPU', varargin{jj})
        GPU = varargin{jj+1};
    elseif strcmp('Version', varargin{jj})
        Version = varargin{jj+1};
    elseif strcmp('LoadResultsVersion', varargin{jj})
        LoadResultsVersion = varargin{jj+1};
    elseif strcmp('SaveFolder', varargin{jj})
        SaveFolder = varargin{jj+1};
    end
end

if not(isfolder(strcat(SaveFolder,"\",Version,"\HHSimContinuous\")))
    mkdir(strcat(SaveFolder,"\",Version,"\HHSimContinuous\"))
end

gpuDevice(GPU)

if isempty(Version) && isempty(LoadResultsVersion)
    fprintf("Version of policy function is not specified!!! \n")
    HHDecision = NaN;
    HHEndowment = NaN;
    return
elseif isempty(LoadResultsVersion)
    fprintf("No Policyfunctions specified! \n")
    fprintf("Use Version provided as LoadResultsVersion! \n")
    LoadResultsVersion = Version;
end
DATA = load(strcat(SaveFolder,"\Results_continuous_",LoadResultsVersion,".mat"),'ValueFunction','Policy','Grid','Para');
VF = DATA.ValueFunction;
Policy = DATA.Policy;
Grid = DATA.Grid;
Para = DATA.Para;

VF_Stay = VF.Stay.Fine;
VF_Mov = VF.Mov.Fine;
V0 = VF.v0;

Emov = Policy.E.Mov;
Amov = Policy.A.Mov;
Astay = Policy.A.Stay;
Wmov = Policy.W.Mov;
Wstay = Policy.W.Stay;
Pmov = Policy.P.Mov;
Pstay = Policy.P.Stay;
Vulmov = Policy.Vul.Mov;
Vulstay = Policy.Vul.Stay;


%% Basic Parameters
delta = Para.delta;         % Wealth depreciation rate
rM = Para.rM; 
rP = Para.rP;
Deltay = Para.Deltay;       % Share of income lost due to disaster
DeltaW = Para.DeltaW;       % Share of wealth lost due to relocation
theta = Para.theta;         % Weight of wealth relative to consumption in utility
beta = Para.beta;           % CES-parameter in utility function
gamma = Para.gamma;         % CRRA-parameter in utility function
scalefactor = Para.scalefactor;  % Scale factor for absolute value of consumption and wealth in utility function
lowC = Para.lowC;           % Minimum consumption level to survive
lowW = Para.lowW;           % Minimum wealth level to survive
kappaZ = Para.kappaZ;       % Share of wealth remaining when liquidating it into financial savings
pFactor = Para.pFactor;


%% Grid Definitions
Egrid = Grid.E;
Agrid = squeeze(Grid.A(1,1,1,1,:))';
Wgrid = squeeze(Grid.W(1,1,1,1,:))';
%Wgrid = squeeze(Grid.W(1,1,1,1,[1,3:end]))';
Dgrid = Grid.D;
Ygrid = Grid.Y;

nD = length(Dgrid);
nE = length(Egrid);
nA = length(Agrid);
nW = length(Wgrid);
nY = size(Ygrid,2);

%-------------------------------------------------------------
ParaBasic = BasicParameters(nY);
ParaBasic_pEfunction = zeros(nE);
Para_ETransfer = zeros(nE);
for iE = 1:nE
    ParaBasic_pEfunction(iE) = ParaBasic.pEfunction(Egrid(iE));
    Para_ETransfer(iE) = Para.ETransfer(Egrid(iE));
end
ParaBasic_pFactor = ParaBasic.pFactor;
%-------------------------------------------------------------

H0grid = Grid.H0;
rhogrid = Grid.rho;
agrid = Grid.a;
phiPgrid = Grid.phiP;

nH0 = length(H0grid);
nrho = length(rhogrid);
na = length(agrid);
nphiP = length(phiPgrid);

TransY_H0 = Para.TransY;
YTransfer_H0 = Grid.YTransfer_H0;
pE = Grid.pE;     

%% ---------------------------------------------------------------------------------------------------

Emov(Emov==0) = Egrid(nE);
Amov(Emov==0) = Agrid(1);
Wmov(Emov==0) = Wgrid(1);
Pmov(Emov==0) = 0;

%% ---------------------------------------------------------------------------------------------------
nHH = 100000;
MaxIterations = 30;
ASmooth = 0.05;
wSmooth = 0.05;

function [Eprime,Aprime,Wprime,Pprime,Iprime,cprime,wprime,...
            Vulprime,NetSavings,Dprime,Yprime,W2prime,...
            EPath,APath,WPath,YPath,DPath,...
            ExpU1,ExpU2,...
            HHpESubsidy,HHYTransfer,HHpETransfer,HHPrevExpSubsidy] = HHSim(iHH)
    
    E = HHEndowment(iHH,1);
    A = HHEndowment(iHH,2);
    W = HHEndowment(iHH,3);
    Y = HHEndowment(iHH,4);
    D = HHEndowment(iHH,5);
    
    Eprime = 0;
    Aprime = 0;
    Wprime = 0;
    Pprime = 0;
    Iprime = 0;
    cprime = 0;
    wprime = 0;
    Vulprime = 0;
    NetSavings = 0;
    Dprime = 0;
    Yprime = 0;
    W2prime = 0;
    
    ExpU1 = 0;
    ExpU2 = 0;
    
    EPath = E;
    APath = A;
    WPath = W;
    YPath = Y;
    DPath = D;    
    
    HHpESubsidy = 0;
    HHYTransfer = 0;
    HHpETransfer = 0;
    HHPrevExpSubsidy = 0;
    
    for iteration = 1:MaxIterations
        jE = 0;
        for kE = 1:nE
            if Egrid(kE) >= E
                jE = kE;
                break
            end
        end
        
        jY = 0;
        for kY = 1:nY
            if Ygrid(iH0,kY) == Y
                jY = kY;
                break
            end
        end
        
        jD = D+1;
        
        jA = 0;
        for kA = 2:nA
            if kA == nA
                jA = nA - 1;
            elseif Agrid(kA) > A
                jA = kA - 1;
                break
            end
        end
        tA = (A - Agrid(jA))/(Agrid(jA+1)-Agrid(jA));
        
        jW = 0;
        for kW = 2:nW
            if kW == nW
                jW = nW - 1;
            elseif Wgrid(kW) > W
                jW = kW - 1;
                break
            end
        end
        tW = (W - Wgrid(jW))/(Wgrid(jW+1)-Wgrid(jW));
    
        VFMOV = (1-tA)*((1-tW)*VF_Mov(iH0,irho,ia,iphiP, jA  ,jW  ,jY,jD)   ...
            +  tW * VF_Mov(iH0,irho,ia,iphiP, jA  ,jW+1,jY,jD))  ...
            + tA *((1-tW)* VF_Mov(iH0,irho,ia,iphiP, jA+1,jW  ,jY,jD)   ...
            +  tW * VF_Mov(iH0,irho,ia,iphiP, jA+1,jW+1,jY,jD));
        
        VFSTAY = (1-tA)*((1-tW)*VF_Stay(iH0,irho,ia,iphiP,jE, jA  ,jW  ,jY,jD)   ...
            +  tW * VF_Stay(iH0,irho,ia,iphiP,jE, jA  ,jW+1,jY,jD))  ...
            + tA *((1-tW)* VF_Stay(iH0,irho,ia,iphiP,jE, jA+1,jW  ,jY,jD)   ...
            +  tW * VF_Stay(iH0,irho,ia,iphiP,jE, jA+1,jW+1,jY,jD));
        
        checkE = 1;
        for kA = jA:jA+1
            for kW = jW:jW+1
                if Emov(iH0,irho,ia,iphiP,jA,jW,jY,jD) ~= Emov(iH0,irho,ia,iphiP,kA,kW,jY,jD)
                    checkE = 0;
                end
            end
        end
        
        if VFSTAY >= VFMOV
            Eprime = E;
            Iprime = 0;
            Aprime = (1-tA)*((1-tW)*Astay(iH0,irho,ia,iphiP,jE, jA  ,jW  ,jY,jD,1)   ...
                +  tW *Astay(iH0,irho,ia,iphiP,jE, jA  ,jW+1,jY,jD,1))  ...
                + tA *((1-tW)*Astay(iH0,irho,ia,iphiP,jE, jA+1,jW  ,jY,jD,1)   ...
                +  tW *Astay(iH0,irho,ia,iphiP,jE, jA+1,jW+1,jY,jD,1));
            Wprime = (1-tA)*((1-tW)*Wstay(iH0,irho,ia,iphiP,jE, jA  ,jW  ,jY,jD,1)   ...
                +  tW *Wstay(iH0,irho,ia,iphiP,jE, jA  ,jW+1,jY,jD,1))  ...
                + tA *((1-tW)*Wstay(iH0,irho,ia,iphiP,jE, jA+1,jW  ,jY,jD,1)   ...
                +  tW *Wstay(iH0,irho,ia,iphiP,jE, jA+1,jW+1,jY,jD,1));
            Pprime = (1-tA)*((1-tW)*Pstay(iH0,irho,ia,iphiP,jE, jA  ,jW  ,jY,jD,1)   ...
                +  tW *Pstay(iH0,irho,ia,iphiP,jE, jA  ,jW+1,jY,jD,1))  ...
                + tA *((1-tW)*Pstay(iH0,irho,ia,iphiP,jE, jA+1,jW  ,jY,jD,1)   ...
                +  tW *Pstay(iH0,irho,ia,iphiP,jE, jA+1,jW+1,jY,jD,1));
            Vulprime = (1-tA)*((1-tW)*Vulstay(iH0,irho,ia,iphiP,jE, jA  ,jW  ,jY,jD,1)   ...
                +  tW *Vulstay(iH0,irho,ia,iphiP,jE, jA  ,jW+1,jY,jD,1))  ...
                + tA *((1-tW)*Vulstay(iH0,irho,ia,iphiP,jE, jA+1,jW  ,jY,jD,1)   ...
                +  tW *Vulstay(iH0,irho,ia,iphiP,jE, jA+1,jW+1,jY,jD,1));
            
            wprime = Wprime - (1-delta)*W;
            if wprime <= -wSmooth
                winvest = kappaZ;
            elseif wprime >= wSmooth
                winvest = 1;
            else
                winvest = kappaZ + ((wprime/wSmooth)/(1+(wprime/wSmooth)^2)+0.5)*(1-kappaZ);
            end
            
            if A >= ASmooth
                TotalIncome = Y*(1-Deltay*D) + (1+rP)*A - pE(jE) + YTransfer(jY);
            elseif A <= -ASmooth
                TotalIncome = Y*(1-Deltay*D) + (1+rM)*A - pE(jE) + YTransfer(jY);
            else
                r = rM + ((A/ASmooth)/(1+(A/ASmooth)^2)+0.5)*(rP-rM);
                TotalIncome = Y*(1-Deltay*D) + (1+r)*A - pE(jE) + YTransfer(jY);
            end
            
            cprime = TotalIncome - wprime*winvest - Aprime - pFactor * Egrid(jE) * Wprime * max(Pprime,0.0)^(1+phiPgrid(iphiP));
            %wprime2 = wprime*winvest;
            
            jE2 = jE;
            jA2 = 0;
            for kA = 2:nA
                if Agrid(kA) > Aprime
                    jA2 = kA - 1;
                    break
                elseif kA == nA
                    jA2 = nA - 1;
                end
            end
            jW2 = 0;
            for kW = 2:nW
                if Wgrid(kW) > Wprime
                    jW2 = kW - 1;
                    break
                elseif kW == nW
                    jW2 = nW-1;
                end
            end
            jP2 = 0;
            for kW = 2:nW
                if Wgrid(kW) > Pprime*Wprime
                    jP2 = kW - 1;
                    break
                elseif kW == nW
                    jP2 = nW-1;
                end
            end
            tP2 = (Pprime*Wprime - Wgrid(jP2))/(Wgrid(jP2+1)-Wgrid(jP2));
            tA2 = (Aprime - Agrid(jA2))/(Agrid(jA2+1)-Agrid(jA2));
            tW2 = (Wprime - Wgrid(jW2))/(Wgrid(jW2+1)-Wgrid(jW2));
                
            ExpU1 = 0;
            ExpU2 = 0;
            for jY2 = 1:nY
                ExpU1 = ExpU1 + TransY_H0(iH0,jY,jY2) * ( ...
                    (1-tA2)*((1-tW2)*V0(iH0,irho,ia,iphiP,jE2, jA2  ,jW2  ,jY2,1)   ...
                             +  tW2 *V0(iH0,irho,ia,iphiP,jE2, jA2  ,jW2+1,jY2,1))  ...
                     + tA2 *((1-tW2)*V0(iH0,irho,ia,iphiP,jE2, jA2+1,jW2  ,jY2,1)   ...
                             +  tW2 *V0(iH0,irho,ia,iphiP,jE2, jA2+1,jW2+1,jY2,1)));
                ExpU2 = ExpU2 + TransY_H0(iH0,jY,jY2) * ( ...
                    (1-tA2)*((1-tP2)*V0(iH0,irho,ia,iphiP,jE2, jA2  ,jP2  ,jY2,2)   ...
                             +  tP2 *V0(iH0,irho,ia,iphiP,jE2, jA2  ,jP2+1,jY2,2))  ...
                     + tA2 *((1-tP2)*V0(iH0,irho,ia,iphiP,jE2, jA2+1,jP2  ,jY2,2)   ...
                             +  tP2 *V0(iH0,irho,ia,iphiP,jE2, jA2+1,jP2+1,jY2,2)));
            end 
        else
            if checkE == 1
                Eprime = Emov(iH0,irho,ia,iphiP,jA,jW,jY,jD);
            else
                if tA <= RelocateHelp(iHH,iteration,1) && tW <= RelocateHelp(iHH,iteration,2)
                    Eprime = Emov(iH0,irho,ia,iphiP,jA,jW,jY,jD);
                elseif tA <= RelocateHelp(iHH,iteration,1) && tW > RelocateHelp(iHH,iteration,2)
                    Eprime = Emov(iH0,irho,ia,iphiP,jA,jW+1,jY,jD);
                elseif tA > RelocateHelp(iHH,iteration,1) && tW <= RelocateHelp(iHH,iteration,2)
                    Eprime = Emov(iH0,irho,ia,iphiP,jA+1,jW,jY,jD);
                else
                    Eprime = Emov(iH0,irho,ia,iphiP,jA+1,jW+1,jY,jD);
                end
            end
            jEprime = 0;
            for kE = 1:nE
                if Egrid(kE) == Eprime
                    jEprime = kE;
                    break
                end
            end
            Iprime = 1;
            Aprime = (1-tA)*((1-tW)*Amov(iH0,irho,ia,iphiP,jA  ,jW  ,jY,jD,1)   ...
                + tW *Amov(iH0,irho,ia,iphiP,jA  ,jW+1,jY,jD,1))  ...
                + tA *((1-tW)*Amov(iH0,irho,ia,iphiP, jA+1,jW  ,jY,jD,1)   ...
                + tW *Amov(iH0,irho,ia,iphiP, jA+1,jW+1,jY,jD,1));
            Wprime = (1-tA)*((1-tW)*Wmov(iH0,irho,ia,iphiP, jA  ,jW  ,jY,jD,1)   ...
                + tW *Wmov(iH0,irho,ia,iphiP, jA  ,jW+1,jY,jD,1))  ...
                + tA *((1-tW)*Wmov(iH0,irho,ia,iphiP, jA+1,jW  ,jY,jD,1)   ...
                + tW *Wmov(iH0,irho,ia,iphiP, jA+1,jW+1,jY,jD,1));
            Pprime = (1-tA)*((1-tW)*Pmov(iH0,irho,ia,iphiP, jA  ,jW  ,jY,jD,1)   ...
                + tW *Pmov(iH0,irho,ia,iphiP, jA  ,jW+1,jY,jD,1))  ...
                + tA *((1-tW)*Pmov(iH0,irho,ia,iphiP, jA+1,jW  ,jY,jD,1)   ...
                + tW *Pmov(iH0,irho,ia,iphiP, jA+1,jW+1,jY,jD,1));
            Vulprime = (1-tA)*((1-tW)*Vulmov(iH0,irho,ia,iphiP, jA  ,jW  ,jY,jD,1)   ...
                + tW *Vulmov(iH0,irho,ia,iphiP, jA  ,jW+1,jY,jD,1))  ...
                + tA *((1-tW)*Vulmov(iH0,irho,ia,iphiP, jA+1,jW  ,jY,jD,1)   ...
                + tW *Vulmov(iH0,irho,ia,iphiP, jA+1,jW+1,jY,jD,1));
            
            if jE == jEprime
                Iprime = 0;
                wprime = Wprime - (1-delta)*W;
            else
                wprime = Wprime - (1-delta)*(1-DeltaW)*W;
            end
                        
            if wprime <= -wSmooth
                winvest = kappaZ;
            elseif wprime >= wSmooth
                winvest = 1;
            else
                winvest = kappaZ + ((wprime/wSmooth)/(1+(wprime/wSmooth)^2)+0.5)*(1-kappaZ);
            end
            %wprime2 = w*winvest;
            
            if A >= ASmooth
                TotalIncome = Y*(1-Deltay*D) + (1+rP)*A - pE(jEprime) + YTransfer(jY);
            elseif A <= -ASmooth
                TotalIncome = Y*(1-Deltay*D) + (1+rM)*A - pE(jEprime) + YTransfer(jY);
            else
                r = rM + ((A/ASmooth)/(1+(A/ASmooth)^2)+0.5)*(rP-rM);
                TotalIncome = Y*(1-Deltay*D) + (1+r)*A - pE(jEprime) + YTransfer(jY);
            end
            
            cprime = TotalIncome - wprime*winvest - Aprime - pFactor * Eprime * Wprime * max(Pprime,0.0)^(1+phiPgrid(iphiP));
            
            jE2 = jEprime;
            jA2 = 0;
            for kA = 2:nA
                if Agrid(kA) > Aprime
                    jA2 = kA - 1;
                    break
                elseif kA == nA
                    jA2 = nA - 1;
                end
            end
            jW2 = 0;
            for kW = 2:nW
                if Wgrid(kW) > Wprime
                    jW2 = kW - 1;
                    break
                elseif kW == nW
                    jW2 = nW-1;
                end
            end
            jP2 = 0;
            for kW = 2:nW
                if Wgrid(kW) > Pprime*Wprime
                    jP2 = kW - 1;
                    break
                elseif kW == nW
                    jP2 = nW-1;
                end
            end
            tP2 = (Pprime*Wprime - Wgrid(jP2))/(Wgrid(jP2+1)-Wgrid(jP2));
            tA2 = (Aprime - Agrid(jA2))/(Agrid(jA2+1)-Agrid(jA2));
            tW2 = (Wprime - Wgrid(jW2))/(Wgrid(jW2+1)-Wgrid(jW2));
                
            ExpU1 = 0;
            ExpU2 = 0;
            for jY2 = 1:nY
                ExpU1 = ExpU1 + TransY_H0(iH0,jY,jY2) * ( ...
                    (1-tA2)*((1-tW2)*V0(iH0,irho,ia,iphiP,jE2, jA2  ,jW2  ,jY2,1)   ...
                             +  tW2 *V0(iH0,irho,ia,iphiP,jE2, jA2  ,jW2+1,jY2,1))  ...
                     + tA2 *((1-tW2)*V0(iH0,irho,ia,iphiP,jE2, jA2+1,jW2  ,jY2,1)   ...
                             +  tW2 *V0(iH0,irho,ia,iphiP,jE2, jA2+1,jW2+1,jY2,1)));
                ExpU2 = ExpU2 + TransY_H0(iH0,jY,jY2) * ( ...
                    (1-tA2)*((1-tP2)*V0(iH0,irho,ia,iphiP,jE2, jA2  ,jP2  ,jY2,2)   ...
                             +  tP2 *V0(iH0,irho,ia,iphiP,jE2, jA2  ,jP2+1,jY2,2))  ...
                     + tA2 *((1-tP2)*V0(iH0,irho,ia,iphiP,jE2, jA2+1,jP2  ,jY2,2)   ...
                             +  tP2 *V0(iH0,irho,ia,iphiP,jE2, jA2+1,jP2+1,jY2,2)));
            end
        end
        
        HHpESubsidy = ParaBasic_pEfunction(jE) - pE(jE);
        HHYTransfer = YTransfer(jY);
        HHpETransfer = Para_ETransfer(jE);
        HHPrevExpSubsidy = (ParaBasic_pFactor - pFactor) * Eprime * Wprime * max(Pprime,0.0)^(1+phiPgrid(iphiP));
        
        NetSavings = Aprime - A;
        
        jYprime = 1;
        TransY_Cum = TransY_H0(iH0,jY,1);
        for kY = 1:nY
            if kY == nY
                jYprime = nY;
            elseif Ysim(iHH,iteration) >= TransY_Cum
                TransY_Cum = TransY_Cum + TransY_H0(iH0,jY,kY + 1);
                jYprime = jYprime + 1;
            else
                break
            end
        end
        Yprime = Ygrid(iH0,jYprime);
        
        if Dsim(iHH,iteration) < Eprime
            Dprime = 1;
        else
            Dprime = 0;
        end
        
        if Dprime == 1
            W2prime = Wprime * Pprime;
        else
            W2prime = Wprime;
        end
        
        EPath = E;
        APath = A;
        WPath = W;
        YPath = Y;
        DPath = D;

        E = Eprime;
        W = W2prime;
        A = Aprime;
        D = Dprime;
        Y = Yprime;
    end
end

nHHGridSim = gpuArray(1:nHH);

HHEndowment = zeros(nHH,5);
HHDecision = zeros(nHH,9);
HHTransfers = zeros(nHH,5);
HHUtility = zeros(nHH,2);

MaxIter = 10;

for iH0 = 1:nH0
    for irho = 1:nrho
        for ia = 1:na
            for iphiP = 1:nphiP
                tic
                
                nquants = 22;
                Pquant = linspace(0.0,0.99,nquants);
                quants = zeros(5,nquants);
                stop = 0;
                iter = 1;
                                                
                Agrid = squeeze(Grid.A(iH0,irho,ia,iphiP,:))';
                Wgrid = squeeze(Grid.W(iH0,irho,ia,iphiP,:))';
                YTransfer = squeeze(YTransfer_H0(iH0,:));
                lowA = Agrid(1);
                lowW = Wgrid(1);
                
                HHEndowment(1:nHH,1) = Egrid(unidrnd(nE,nHH,1));
                HHEndowment(1:nHH,2) = lowA + rand(nHH,1)*(Agrid(end)-lowA);
                HHEndowment(1:nHH,3) = lowW + rand(nHH,1)*(Wgrid(end)-lowW);
                HHEndowment(1:nHH,4) = reshape(Ygrid(iH0,unidrnd(nY,nHH,1)),1,nHH,1);
                HHEndowment(1:nHH,5) = rand(nHH,1) < HHEndowment(1:nHH,1);
                    
                while stop == 0 && iter <= MaxIter
                    Dsim = rand(nHH,MaxIterations);
                    Ysim = rand(nHH,MaxIterations);
                    RelocateHelp = rand(nHH,MaxIterations,2);
                    
                    quantsOld = quants;
                    [Eprime,Aprime,Wprime,Pprime,Iprime,cprime,wprime,...
                        Vulprime,NetSavings,Dprime,Yprime,W2prime,...
                        EPath,APath,WPath,YPath,DPath,...
                        ExpU1,ExpU2,...
                        HHpESubsidy,HHYTransfer,HHpETransfer,HHPrevExpSubsidy] = arrayfun(@HHSim,nHHGridSim);
                    
                    HHEndowment(:,1) = EPath;
                    HHEndowment(:,2) = APath;
                    HHEndowment(:,3) = WPath;
                    HHEndowment(:,4) = YPath;
                    HHEndowment(:,5) = DPath; 
                    HHDecision(:,1) = Eprime;
                    HHDecision(:,2) = Aprime;
                    HHDecision(:,3) = Wprime;
                    HHDecision(:,4) = Pprime;
                    HHDecision(:,5) = Iprime;
                    HHDecision(:,6) = cprime;
                    HHDecision(:,7) = wprime;
                    HHDecision(:,8) = Vulprime;
                    HHDecision(:,9) = NetSavings;
                    HHTransfers(:,1) = HHpESubsidy;
                    HHTransfers(:,2) = HHYTransfer;
                    HHTransfers(:,3) = HHpETransfer;
                    HHTransfers(:,4) = HHPrevExpSubsidy;
                    HHUtility(:,1) = ExpU1;
                    HHUtility(:,2) = ExpU2;
                                        
                    TitleVariables = ['E' 'A' 'W' 'P' 'I'];
                    for jj = 1:5
                        quants(jj,:) = quantile(HHDecision(:,jj),Pquant);
                        subplot(3,2,jj)
                        title(TitleVariables(jj))
                        plot(quants(jj,:),Pquant)
                        hold on
                    end
                    sgtitle(strcat("H0=",num2str(iH0)," | rho=",num2str(irho)," | a=",num2str(ia)," | phiP=",num2str(iphiP),"| iter=",num2str(iter)))
                    pause(0.001)
                    
                    DiffQuantiles = max(abs(quants-quantsOld),[],2);
                    fprintf("Maximum differences of quantiles:")
                    fprintf("E = %g | A = %g | W = %g | P = %g | I = %g \n",DiffQuantiles(1),DiffQuantiles(2),DiffQuantiles(3),DiffQuantiles(4),DiffQuantiles(5))
                    if all(max(abs(quants-quantsOld),[],2) < [1e-3;Agrid(end)*1e-3;Wgrid(end)*1e-3;1e-3;1e-3]) || iter >= MaxIter
                        stop = 1;
                        for jj = 1:5
                            subplot(3,2,jj)
                            hold off
                        end
                        Dec.E = HHDecision(:,1);
                        Dec.A = HHDecision(:,2);
                        Dec.W = HHDecision(:,3);
                        Dec.P = HHDecision(:,4);
                        Dec.I = HHDecision(:,5);
                        Dec.c = HHDecision(:,6);
                        Dec.w = HHDecision(:,7);
                        Dec.Vul = HHDecision(:,8);
                        Dec.NS = HHDecision(:,9);
                        
                        Path.E = HHEndowment(:,1);
                        Path.A = HHEndowment(:,2);
                        Path.W = HHEndowment(:,3);
                        Path.Y = HHEndowment(:,4);
                        Path.D = HHEndowment(:,5);
                        
                        Characteristics.H0 = H0grid(iH0);
                        Characteristics.a = agrid(ia);
                        Characteristics.rho = rhogrid(irho);
                        Characteristics.phiP = phiPgrid(iphiP);
                        Policies.pESubsidy = HHpESubsidy;
                        Policies.YTransfer = HHYTransfer;
                        Policies.pETransfer = HHpETransfer;
                        Policies.PrevExpSubsidy = HHPrevExpSubsidy;
                        save(strcat(SaveFolder,"\",Version,"\HHSimContinuous\Part_",num2str(iH0),"_",num2str(irho),"_",num2str(ia),"_",num2str(iphiP),".mat"),'Dec','Path','Characteristics','Grid','Policies','HHUtility','Para', '-v7.3')
                    else
                        iter = iter+1;
                    end
                end
                fprintf("Combination iH0 = %d | irho = %d | ia = %d | iphiP = %d | time = %g | iter = %d\n",iH0,irho,ia,iphiP,toc,iter)
                fprintf("--------------------------------------\n")
            end
        end
    end
end

end
