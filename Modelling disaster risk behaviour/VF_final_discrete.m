
function [ValueFunction,Policy,Grid] = VF_final_discrete(varargin)

% DefaultValues
GPU = 1;
Gridsizes = [2,2,2,2,2];
Version = 'Test';
ParaPol = struct();
LoadResultsVersion = '';
SaveFolder = "D:\Users\mfreiber\DisasterRiskModel\Matlab-Simulations_V2.1";

for jj = 1:2:nargin
    if strcmp('GPU', varargin{jj})
        GPU = varargin{jj+1};
    elseif strcmp('Gridsizes', varargin{jj})
        Gridsizes = varargin{jj+1};
    elseif strcmp('Version', varargin{jj})
        Version = varargin{jj+1};
    elseif strcmp('ParaPol', varargin{jj})
        ParaPol = varargin{jj+1};
    elseif strcmp('LoadResultsVersion', varargin{jj})
        LoadResultsVersion = varargin{jj+1};
    elseif strcmp('SaveFolder', varargin{jj})
        SaveFolder = varargin{jj+1};
    end
end

if not(isfolder(SaveFolder))
    mkdir(SaveFolder)
end

gpuDevice(GPU)
totaltime = tic;

Gridsizes = num2cell(Gridsizes);
[nE,nA,nW,nP,nY] = Gridsizes{1,:};

%% ---------------------------------------------------------------------------------------------------------------------
%   Load initial data (optional)
%-----------------------------------------------------------------------------------------------------------------------

if isempty(LoadResultsVersion)
    Para = BasicParameters(nY);
else
    DATA = load(strcat(SaveFolder,"\Results_discrete_",LoadResultsVersion,".mat"),'ValueFunction','Policy','Grid','Para');
    ValueFunctionOld = DATA.ValueFunction;
    PolicyOld = DATA.Policy;
    GridOld = DATA.Grid;
    Para = DATA.Para;
    
    EgridOld = GridOld.E;
    AgridOld = squeeze(GridOld.A(1,1,1,1,:))';
    WgridOld = squeeze(GridOld.W(1,1,1,1,:))';
    PgridOld = GridOld.P;
    YgridOld = GridOld.Y(1,:);
    
    if nE == 2 && nA == 2 && nW == 2 && nP == 2 && nY == 2
        nE = length(EgridOld);
        nA = length(AgridOld);
        nW = length(WgridOld);
        nP = length(PgridOld);
        nY = length(YgridOld);
    end   
end


fn = fieldnames(ParaPol);
for k=1:numel(fn)
    if isfield(Para,fn{k})
        Para.(strcat(fn{k},'Base')) = Para.(fn{k});
    end
    Para.(fn{k}) = ParaPol.(fn{k});
end
Para.pEfunction = @(E) Para.HousingMin*exp(Para.HousingCoef*(1-E)^Para.HousingPower);


delta = Para.delta;                 % Wealth depreciation raterate
rP = Para.rP;                       % Interest rate for positive savings
rM = Para.rM;                       % Interest rate for debt
Deltay = Para.Deltay;               % Share of income lost due to disaster
DeltaW = Para.DeltaW;               % Share of wealth lost due to relocation
theta = Para.theta;                 % Weight of wealth relative to consumption in utility
beta = Para.beta;                   % CES-parameter in utility function
gamma = Para.gamma;                 % CRRA-parameter in utility function
scalefactor = Para.scalefactor;     % Scale factor for absolute value of consumption and wealth in utility function
lowC = Para.lowC;                   % Minimum consumption level to survive
lowW = Para.lowW;                   % Minimum wealth level to survive
kappaZ = Para.kappaZ;               % Share of wealth remaining when liquidating it into financial savings
pFactor = Para.pFactor;

%% Grid Definitions
nD = 2;
nH0 = Para.nH0;
nrho = Para.nrho;
na = Para.na;
nphiP = Para.nphiP;

[Egrid,Agrid,Wgrid,Dgrid,Pgrid,Ygrid_H0,TransY_H0,YTransfer_H0,H0grid,rhogrid,agrid,phiPgrid] = CreateGrids(nE,nA,nW,nY,nD,nP,Para);

indexW = find(Wgrid >lowW,1,'first');
Para.TransY = TransY_H0;

Grid.E = Egrid;
Grid.A = zeros(nH0,nrho,na,nphiP,nA);
Grid.Abase = Agrid;
Grid.W = zeros(nH0,nrho,na,nphiP,nW);
Grid.Wbase = Wgrid;
Grid.P = Pgrid;
Grid.D = Dgrid;
Grid.Y = Ygrid_H0;
Grid.H0 = H0grid;
Grid.rho = rhogrid;
Grid.rho2 = 1./(1 + Grid.rho);
Grid.a = agrid;
Grid.phiP = phiPgrid;
Grid.TransY = Para.TransY;
Grid.YTransfer_H0 = YTransfer_H0;

[Agrid4,Wgrid4,Ygrid4,Dgrid4] = ndgrid(gpuArray(1:nA),gpuArray(1:nW),gpuArray(1:nY),gpuArray(1:nD) );
%[Agrid4,Wgrid4,Ygrid4,Dgrid4] = ndgrid(1:nA,1:nW,1:nY,1:nD);    
[Egrid5,Agrid5,Wgrid5,Ygrid5,Dgrid5] = ndgrid( gpuArray(1:nE), gpuArray(1:nA),gpuArray(1:nW),gpuArray(1:nY),gpuArray(1:nD) );
%[Egrid5_V2,Agrid5_V2,Wgrid5_V2,Ygrid5_V2,Dgrid5_V2] = ndgrid(1:nE,1:nA,1:nW,1:nY,1:nD);    


%% -----------------------------------------------------------------------------------------------------------------------
%   Initialize Variables
%-----------------------------------------------------------------------------------------------------------------------
v0 = zeros(nE,nA,nW,nY,nD);
v0Dis = zeros(nE,nA,nW,nY,nD);
v2Mov = zeros(nA,nW,nY,nD);
v2Stay = zeros(nE,nA,nW,nY,nD);

PolicyE = ones(nE,nA,nW,nY,nD);
PolicyI = ones(nE,nA,nW,nY,nD);
PolicyA = ones(nE,nA,nW,nY,nD);
PolicyW = ones(nE,nA,nW,nY,nD);
PolicyP = ones(nE,nA,nW,nY,nD);

PolicyMovE = nE*ones(nA,nW,nY,nD);
PolicyMovA = ones(nA,nW,nY,nD);
PolicyMovW = indexW*ones(nA,nW,nY,nD);
PolicyMovP = nP*ones(nA,nW,nY,nD);

PolAMovHelp = ones(nE,nA,nW,nY,nD);
PolWMovHelp = indexW*ones(nE,nA,nW,nY,nD);
PolPMovHelp = ones(nE,nA,nW,nY,nD);

PolicyStayA = ones(nE,nA,nW,nY,nD);
PolicyStayW = indexW*ones(nE,nA,nW,nY,nD);
PolicyStayP = nP*ones(nE,nA,nW,nY,nD);
   

%% ---------------------------------------------------------------------------------------------------------------------
%   Help Functions
%-----------------------------------------------------------------------------------------------------------------------
function [w,w2] = PreCalculations(jW,iW,jI)
    if jI == 1
        w = Wgrid(jW) - (1-delta)*(1-DeltaW)*Wgrid(iW);
    else
        w = Wgrid(jW) - (1-delta)*Wgrid(iW);
    end
    if w < 0
        w2 = w*kappaZ;
    else
        w2 = w;
    end
end

function [UWPart] = UW(jW)
    if Wgrid(jW) > lowW
        UWPart = theta * ( ((Wgrid(jW) -lowW)/scalefactor)^(1-beta) - 1)/(1-beta);
    else
        UWPart = -1e9;
    end
end

function [kP,t] = FindPindex(jW,jP)
    kP=1;
    stop=0;
    while stop == 0
        if Wgrid(jW)*Pgrid(jP) > Wgrid(kP)
            kP = kP + 1;
        else
            stop = 1;
            kP = max(1,kP - 1);
        end
    end
    t = (Wgrid(jW)*Pgrid(jP) - Wgrid(kP))/(Wgrid(kP+1)-Wgrid(kP));
end

pE = gpuArray(zeros(1,nE));
for iE = 1:nE
    pE(iE) = Para.pEfunction(Egrid(iE)) - Para.ETransfer(Egrid(iE));
end
Grid.pE = pE;

%% ---------------------------------------------------------------------------------------------------------------------
%   Basic Value-Function-Iterations
%-----------------------------------------------------------------------------------------------------------------------

function [v1,PolA,PolW,PolP] = OptiSearchMov(jEstar,iA,iW,iY,iD) 

    jAstar = PolAMovHelp(jEstar,iA,iW,iY,iD);
    jWstar = PolWMovHelp(jEstar,iA,iW,iY,iD);
    jPstar = PolPMovHelp(jEstar,iA,iW,iY,iD);

    E0 = Egrid(jEstar);
    A0 = Agrid(iA);
    W0 = Wgrid(iW);
    D0 = Dgrid(iD);
    Y0 = Ygrid(iY);

    stopOuter = 0;
    OptiIter = 0;
    FmaxOld = -1e9;
    Fmax = -1e9;
    FmaxOuter = -1e9;
    
    jAstarOuter = 1;
    jPstarOuter = 1;
    jWstarOuter = 1;    
    
    if A0 > 0
        TotalIncome = Y0*(1-Deltay*D0) + (1+rP)*A0 - pE(jEstar) + YTransfer(iY);
    else
        TotalIncome = Y0*(1-Deltay*D0) + (1+rM)*A0 - pE(jEstar) + YTransfer(iY);
    end
    if TotalIncome - w2Matrix(indexW,iW,1) - Agrid(1) - lowC < 0
        stopOuter = 1;
    else    
        %-----------------------------------------------------------------------------------------------------------------------------------------
        %   Search through all W values
        for jW = indexW:nW
            cW = TotalIncome - w2Matrix(jW,iW,1);
            if cW - Agrid(1) <= lowC
                break
            else
                Fmax = -1e9;
                stopOuter = 0;
                OptiIter = 0;
                while stopOuter == 0 && OptiIter < 200

                    jAold = jAstar;
                    jPold = jPstar;
                    OptiIter = OptiIter+1;

                    %-----------------------------------------------------------------------------------------------------------------------------------------
                    %   Search for A
                    cABase = cW - pFactor * E0 * Wgrid(jW) * Pgrid(jPstar)^(1+phiP);
                    uhelp1 = -1e9;
                    for jA = 1:nA
                        cA = cABase - Agrid(jA);
                        if cA <= lowC
                            if jA == 1
                                jAstar = 1;
                            end    
                            break
                        else
                            V0C1 = 0;
                            V0C2 = 0;
                            for jY = 1:nY
                                V0C1 = V0C1 + TransY(iY,jY) * v0(jEstar,jA,jW,jY,1);
                                V0C2 = V0C2 + TransY(iY,jY) * ((1-tWP(jW,jPstar)) * v0(jEstar,jA,kP(jW,jPstar),jY,2) + ...
                                    tWP(jW,jPstar)*v0(jEstar,jA,kP(jW,jPstar)+1,jY,2));
                            end
                            uhelp2 = (((cA-lowC)/scalefactor)^(1-gamma) - 1)/(1-gamma) + UWPart(jW) + ...
                                            + rho2 * (a*E0*V0C2 + (1-a*E0)*V0C1);
                            if uhelp2 > Fmax
                                Fmax = uhelp2;
                                jAstar = jA;
                            end
                            if uhelp2 >= uhelp1
                                uhelp1 = uhelp2;
                            else
                                %break
                            end
                        end
                    end

                    %-----------------------------------------------------------------------------------------------------------------------------------------
                    %   Search for P
                    cPBase = cW - Agrid(jAstar);
                    V0C1 = 0;
                    for jY = 1:nY
                        V0C1 = V0C1 + TransY(iY,jY) * v0(jEstar,jAstar,jW,jY,1);
                    end

                    for jP = 1:nP
                        cP = cPBase - pFactor * E0 * Wgrid(jW) * Pgrid(jP)^(1+phiP);
                        if cP <= lowC
                            if jP == 1
                                jPstar = 1;
                            end
                            break
                        else
                            V0C2 = 0;
                            for jY = 1:nY
                                V0C2 = V0C2 + TransY(iY,jY) * ((1-tWP(jW,jP)) * v0(jEstar,jAstar,kP(jW,jP),jY,2) + ...
                                    + tWP(jW,jP)*v0(jEstar,jAstar,kP(jW,jP)+1,jY,2));
                            end
                            uhelp = (((cP-lowC)/scalefactor)^(1-gamma) - 1)/(1-gamma) + UWPart(jW) + ...
                                + rho2 * (a*E0*V0C2 + (1-a*E0)*V0C1);
                            if uhelp > Fmax
                                Fmax = uhelp;
                                jPstar = jP;
                            end
                        end
                    end
                    if jAstar == jAold && jPstar == jPold
                        stopOuter=1;
                        FmaxOld = Fmax;
                        for jA = max(jAold-3,1):min(jAold+3,nA)
                            for jP = 1:nP%max(1,jPold-5):min(jPold+5,nP)
                                if (jA ~= jAold) + (jP ~= jPold) >= 2
                                    c = cW - Agrid(jA) - pFactor * E0 * Wgrid(jW) * Pgrid(jP)^(1+phiP);
                                    if c <= lowC
                                        unew = -1e9;
                                    else
                                        V0C1 = 0;
                                        V0C2 = 0;
                                        for jY = 1:nY
                                            V0C1 = V0C1 + TransY(iY,jY) * v0(jEstar,jA,jW,jY,1);
                                            V0C2 = V0C2 + TransY(iY,jY) * ((1-tWP(jW,jP)) * v0(jEstar,jA,kP(jW,jP),jY,2) + tWP(jW,jP)*v0(jEstar,jA,kP(jW,jP)+1,jY,2));
                                        end
                                        unew = (((c-lowC)/scalefactor)^(1-gamma) - 1)/(1-gamma) + UWPart(jW) + ...
                                            + rho2 * (a*E0*V0C2 + (1-a*E0)*V0C1);
                                    end
                                    if unew > Fmax
                                       Fmax = unew;
                                       jAstar = jA;
                                       jPstar = jP;
                                    end
                                end
                            end
                        end
                
                        if Fmax > FmaxOld
                            stopOuter = 0;
                        end
                    end
                end
                if Fmax >= FmaxOuter
                    jAstarOuter = jAstar;
                    jPstarOuter = jPstar;
                    jWstarOuter = jW;
                    FmaxOuter = Fmax;
                end
            end
        end
        
    end
    if FmaxOuter == -1e9
        v1 = Fmax;
        PolA = 1;
        PolW = 1;
        PolP = 1;
    else
        v1 = FmaxOuter;
        PolA = jAstarOuter;
        PolW = jWstarOuter;
        PolP = jPstarOuter;
    end          
end

function [v1Mov,PolE,PolA,PolW,PolP] = OptiSearchMovCompare(iA,iW,iY,iD)
   jEmax = 1;
   Vbest = v1MovHelp(1,iA,iW,iY,iD);
   
   for ii = 2:nE
       if v1MovHelp(ii,iA,iW,iY,iD) > Vbest
           jEmax = ii;
           Vbest = v1MovHelp(ii,iA,iW,iY,iD);
       end
   end
   
   v1Mov = v1MovHelp(jEmax,iA,iW,iY,iD);
   PolE = jEmax;
   PolA = PolAMovHelp(jEmax,iA,iW,iY,iD);
   PolW = PolWMovHelp(jEmax,iA,iW,iY,iD);
   PolP = PolPMovHelp(jEmax,iA,iW,iY,iD);
end

function [v1,PolA,PolW,PolP] = OptiSearchMovAllReduced(jEstar,iA,iW,iY,iD) 
    % Search complete grid within +-range of the so-far-found optimal point (and search all prevention levels)
    % (when moving)
    range = 5000;%floor(nE/1);
    A0 = Agrid(iA);
    W0 = Wgrid(iW);
    D0 = Dgrid(iD);
    Y0 = Ygrid(iY);
    
    jAold = PolAMovHelp(jEstar,iA,iW,iY,iD);
    jWold = PolWMovHelp(jEstar,iA,iW,iY,iD);
    jPold = PolPMovHelp(jEstar,iA,iW,iY,iD);    
    
    if A0 > 0
        TotalIncome = Y0*(1-Deltay*D0) + (1+rP)*A0 - pE(jEstar) + YTransfer(iY);
    else
        TotalIncome = Y0*(1-Deltay*D0) + (1+rM)*A0 - pE(jEstar) + YTransfer(iY);
    end
    
    ustar = -1e9;
    jWstar = 1;
    jAstar = 1;
    jPstar = 1;
    
    for jW = max(indexW,jWold-range):min(nW,jWold+range)
        cW = TotalIncome - w2Matrix(jW,iW,1);
        if cW <= lowC + Agrid(1)
            break
        else
            for jA = max(1,jAold-range):min(nA,jAold+range)
                cWA = cW - Agrid(jA);
                if cWA <= lowC
                    break
                else
                    for jP = max(1,jPold-range):min(nP,jPold+range)
                        c = cWA - pFactor * Egrid(jEstar) * Wgrid(jW) * Pgrid(jP)^(1+phiP);
                        if c <= lowC
                            break
                        else
                            V0C1 = 0;
                            V0C2 = 0;
                            for jY = 1:nY
                                V0C1 = V0C1 + TransY(iY,jY) * v0(jEstar,jA,jW,jY,1);
                                V0C2 = V0C2 + TransY(iY,jY) * ((1-tWP(jW,jP)) * v0(jEstar,jA,kP(jW,jP),jY,2) + ...
                                    tWP(jW,jP)*v0(jEstar,jA,kP(jW,jP)+1,jY,2));
                            end
                            uhelp = (((c-lowC)/scalefactor)^(1-gamma) - 1)/(1-gamma) + UWPart(jW) + ...
                                + rho2 * (a*Egrid(jEstar)*V0C2 + (1-a*Egrid(jEstar))*V0C1);
                            if uhelp > ustar
                                ustar = uhelp;
                                jAstar = jA;
                                jWstar = jW;
                                jPstar = jP;
                            end
                        end
                    end
                end
            end
        end
    end
    
    if ustar == -1e9
        v1 = ustar;
        PolA = 1;
        PolW = 1;
        PolP = 1;
    else
        v1 = ustar;
        PolA = jAstar;
        PolW = jWstar;
        PolP = jPstar;
    end
end

function [v1,PolA,PolW,PolP] = OptiSearchStay(iE,iA,iW,iY,iD) 

    jAstar = PolicyStayA(iE,iA,iW,iY,iD);
    jWstar = PolicyStayW(iE,iA,iW,iY,iD);
    jPstar = PolicyStayP(iE,iA,iW,iY,iD);

    E0 = Egrid(iE);
    A0 = Agrid(iA);
    W0 = Wgrid(iW);
    D0 = Dgrid(iD);
    Y0 = Ygrid(iY);
    
    stopOuter = 0;
    OptiIter = 0;
    FmaxOld = -1e9;
    Fmax = -1e9;
    FmaxOuter = -1e9;
    
    jAstarOuter = 1;
    jPstarOuter = 1;
    jWstarOuter = 1;
    
    if A0 > 0
        TotalIncome = Y0*(1-Deltay*D0) + (1+rP)*A0 - pE(iE) + YTransfer(iY);
    else
        TotalIncome = Y0*(1-Deltay*D0) + (1+rM)*A0 - pE(iE) + YTransfer(iY);
    end
    if TotalIncome - w2Matrix(indexW,iW,2) - Agrid(1) - lowC < 0
        stopOuter = 1;
    end        
    
    %-----------------------------------------------------------------------------------------------------------------------------------------
    %   Search through all W values
    for jW = indexW:nW
        cW = TotalIncome - w2Matrix(jW,iW,2);
        if cW - Agrid(1) <= lowC
            break
        else
            Fmax = -1e9;
            stopOuter = 0;
            OptiIter = 0;
            while stopOuter == 0 && OptiIter < 200
                
                jAold = jAstar;
                jPold = jPstar;
                OptiIter = OptiIter+1;
                                
                %-----------------------------------------------------------------------------------------------------------------------------------------
                %   Search for A
                cABase = cW - pFactor * E0 * Wgrid(jW) * Pgrid(jPstar)^(1+phiP);
                uhelp1 = -1e9;
                for jA = 1:nA
                    cA = cABase - Agrid(jA);
                    if cA <= lowC
                        if jA == 1
                            jAstar = 1;
                        end    
                        break
                    else
                        V0C1 = 0;
                        V0C2 = 0;
                        for jY = 1:nY
                            V0C1 = V0C1 + TransY(iY,jY) * v0(iE,jA,jW,jY,1);
                            V0C2 = V0C2 + TransY(iY,jY) * ((1-tWP(jW,jPstar)) * v0(iE,jA,kP(jW,jPstar),jY,2) + ...
                                tWP(jW,jPstar)*v0(iE,jA,kP(jW,jPstar)+1,jY,2));
                        end
                        uhelp2 = (((cA-lowC)/scalefactor)^(1-gamma) - 1)/(1-gamma) + UWPart(jW) + ...
                                        + rho2 * (a*E0*V0C2 + (1-a*E0)*V0C1);
                        if uhelp2 > Fmax
                            Fmax = uhelp2;
                            jAstar = jA;
                        end
                        if uhelp2 >= uhelp1
                            uhelp1 = uhelp2;
                        else
                            %break
                        end
                    end
                end
                
                %-----------------------------------------------------------------------------------------------------------------------------------------
                %   Search for P
                cPBase = cW - Agrid(jAstar);
                V0C1 = 0;
                for jY = 1:nY
                    V0C1 = V0C1 + TransY(iY,jY) * v0(iE,jAstar,jW,jY,1);
                end
                            
                for jP = 1:nP
                    cP = cPBase - pFactor * E0 * Wgrid(jW) * Pgrid(jP)^(1+phiP);
                    if cP <= lowC
                        if jP == 1
                            jPstar = 1;
                        end
                        break
                    else
                        V0C2 = 0;
                        for jY = 1:nY
                            V0C2 = V0C2 + TransY(iY,jY) * ((1-tWP(jW,jP)) * v0(iE,jAstar,kP(jW,jP),jY,2) + ...
                                + tWP(jW,jP)*v0(iE,jAstar,kP(jW,jP)+1,jY,2));
                        end
                        uhelp = (((cP-lowC)/scalefactor)^(1-gamma) - 1)/(1-gamma) + UWPart(jW) + ...
                            + rho2 * (a*E0*V0C2 + (1-a*E0)*V0C1);
                        if uhelp > Fmax
                            Fmax = uhelp;
                            jPstar = jP;
                        end
                    end
                end
                if jAstar == jAold && jPstar == jPold
                    stopOuter=1;
                    if jAstar == jAold && jPstar == jPold
                        stopOuter=1;
                        FmaxOld = Fmax;
                        for jA = max(jAold-3,1):min(jAold+3,nA)
                            for jP = 1:nP %max(1,jPold-5):min(jPold+5,nP)
                                if (jA ~= jAold) + (jP ~= jPold) >= 2
                                    c = cW - Agrid(jA) - pFactor * E0 * Wgrid(jW) * Pgrid(jP)^(1+phiP);
                                    if c <= lowC
                                        unew = -1e9;
                                    else
                                        V0C1 = 0;
                                        V0C2 = 0;
                                        for jY = 1:nY
                                            V0C1 = V0C1 + TransY(iY,jY) * v0(iE,jA,jW,jY,1);
                                            V0C2 = V0C2 + TransY(iY,jY) * ((1-tWP(jW,jP)) * v0(iE,jA,kP(jW,jP),jY,2) + tWP(jW,jP)*v0(iE,jA,kP(jW,jP)+1,jY,2));
                                        end
                                        unew = (((c-lowC)/scalefactor)^(1-gamma) - 1)/(1-gamma) + UWPart(jW) + ...
                                            + rho2 * (a*E0*V0C2 + (1-a*E0)*V0C1);
                                    end
                                    if unew > Fmax
                                       Fmax = unew;
                                       jAstar = jA;
                                       jPstar = jP;
                                    end
                                end
                            end
                        end
                
                        if Fmax > FmaxOld
                            stopOuter = 0;
                        end
                    end
                end
            end
            if Fmax >= FmaxOuter
                jAstarOuter = jAstar;
                jPstarOuter = jPstar;
                jWstarOuter = jW;
                FmaxOuter = Fmax;
            end
        end
    end
        
    if FmaxOuter == -1e9
        v1 = Fmax;
        PolA = 1;
        PolW = 1;
        PolP = 1;
    else
        v1 = FmaxOuter;
        PolA = jAstarOuter;
        PolW = jWstarOuter;
        PolP = jPstarOuter;
    end
end

function [v1,PolA,PolW,PolP] = OptiSearchStayAllReduced(iE,iA,iW,iY,iD) 
    % Search complete grid within +-range of the so-far-found optimal point (and search all prevention levels)
    % (when staying)
    range = 5000;%floor(nE/2);
    
    E0 = Egrid(iE);
    A0 = Agrid(iA);
    W0 = Wgrid(iW);
    D0 = Dgrid(iD);
    Y0 = Ygrid(iY);
    
    jAold = PolicyStayA(iE,iA,iW,iY,iD);
    jWold = PolicyStayW(iE,iA,iW,iY,iD);
    jPold = PolicyStayP(iE,iA,iW,iY,iD);
    
    if A0 > 0
        TotalIncome = Y0*(1-Deltay*D0) + (1+rP)*A0 - pE(iE) + YTransfer(iY);
    else
        TotalIncome = Y0*(1-Deltay*D0) + (1+rM)*A0 - pE(iE) + YTransfer(iY);
    end
    
    ustar = -1e9;
    jWstar = 1;
    jAstar = 1;
    jPstar = 1;
    
    if TotalIncome - w2Matrix(indexW,iW,2) - Agrid(1) - lowC <= 0
        v1 = ustar;
        PolA = 1;
        PolW = 1;
        PolP = 1;
        return
    end
    
    for jW = max(indexW,jWold-range):min(nW,jWold+range)
        cW = TotalIncome - w2Matrix(jW,iW,2);
        if cW <= lowC + Agrid(1)
            break
        else
            for jA = max(1,jAold-range):min(nA,jAold+range)
                cWA = cW - Agrid(jA);
                if cWA <= lowC
                    break
                else
                    for jP = max(1,jPold-range):min(nP,jPold+range)
                        c = cWA - pFactor * E0 * Wgrid(jW) * Pgrid(jP)^(1+phiP);
                        if c <= lowC
                            break
                        else
                            V0C1 = 0;
                            V0C2 = 0;
                            for jY = 1:nY
                                V0C1 = V0C1 + TransY(iY,jY) * v0(iE,jA,jW,jY,1);
                                V0C2 = V0C2 + TransY(iY,jY) * ((1-tWP(jW,jP)) * v0(iE,jA,kP(jW,jP)  ,jY,2) + ...
                                    tWP(jW,jP)  * v0(iE,jA,kP(jW,jP)+1,jY,2));
                            end
                            uhelp = (((c-lowC)/scalefactor)^(1-gamma) - 1)/(1-gamma) + UWPart(jW) + ...
                                + rho2 * (a*E0*V0C2 + (1-a*E0)*V0C1);
                            if uhelp > ustar
                                ustar = uhelp;
                                jAstar = jA;
                                jWstar = jW;
                                jPstar = jP;
                            end
                        end
                    end
                end
            end
        end
    end
    
    if ustar == -1e9
        v1 = ustar;
        PolA = 1;
        PolW = 1;
        PolP = 1;
    else
        v1 = ustar;
        PolA = jAstar;
        PolW = jWstar;
        PolP = jPstar;
    end
end

function [v1,PolE,PolA,PolW,PolP,PolI] = ValueCompare(iE,iA,iW,iY,iD)
    if v1Mov(iA,iW,iY,iD) > v1Stay(iE,iA,iW,iY,iD)
        v1 = v1Mov(iA,iW,iY,iD);
        PolE = PolicyMovE(iA,iW,iY,iD);
        PolA = PolicyMovA(iA,iW,iY,iD);
        PolW = PolicyMovW(iA,iW,iY,iD);
        PolP = PolicyMovP(iA,iW,iY,iD);
        if PolicyMovE(iA,iW,iY,iD) == iE
            PolI = 0;
        else
            PolI = 1;
        end
    else
        v1 = v1Stay(iE,iA,iW,iY,iD);
        PolE = iE;
        PolA = PolicyStayA(iE,iA,iW,iY,iD);
        PolW = PolicyStayW(iE,iA,iW,iY,iD);
        PolP = PolicyStayP(iE,iA,iW,iY,iD);
        PolI = 0;
    end
end

%% ---------------------------------------------------------------------------------------------------------------------
%   Policy Iteration Functions
%-----------------------------------------------------------------------------------------------------------------------

function v2 = PolicyIterationMov(iA,iW,iY,iD) 
    jE = PolicyMovE(iA,iW,iY,iD);
    jA = PolicyMovA(iA,iW,iY,iD);
    jW = PolicyMovW(iA,iW,iY,iD);
    jP = PolicyMovP(iA,iW,iY,iD);
        
    if Agrid(iA) > 0
        Ahelp = (1+rP)*Agrid(iA);
    else
        Ahelp = (1+rM)*Agrid(iA);
    end
    c = Ygrid(iY)*(1-Deltay*Dgrid(iD)) + Ahelp - Agrid(jA) - w2Matrix(jW,iW,1) - pE(jE)  + YTransfer(iY) - ...
                        pFactor * Egrid(jE) * Wgrid(jW) * Pgrid(jP)^(1+phiP);
    
    if c <= lowC || Wgrid(jW) <= lowW
        v2 = -1e9;
    else        
        V0C1 = 0;
        V0C2 = 0;
        for jY = 1:nY
            V0C1 = V0C1 + TransY(iY,jY) * v1(jE,jA,jW,jY,1);
            V0C2 = V0C2 + TransY(iY,jY)*((1-tWP(jW,jP)) * v1(jE,jA,kP(jW,jP),jY,2) + tWP(jW,jP)*v1(jE,jA,kP(jW,jP)+1,jY,2));
        end
        v2 = (((c-lowC)/scalefactor)^(1-gamma) - 1)/(1-gamma) + UWPart(jW) + ...
            + rho2 * (a*Egrid(jE)*V0C2 + (1-a*Egrid(jE))*V0C1);
    end
end

function v2 = PolicyIterationStay(iE,iA,iW,iY,iD) 
    jA = PolicyStayA(iE,iA,iW,iY,iD);
    jW = PolicyStayW(iE,iA,iW,iY,iD);
    jP = PolicyStayP(iE,iA,iW,iY,iD);
        
    if Agrid(iA) > 0
        Ahelp = (1+rP)*Agrid(iA);
    else
        Ahelp = (1+rM)*Agrid(iA);
    end
    
    c = Ygrid(iY)*(1-Deltay*Dgrid(iD)) + Ahelp - Agrid(jA) - w2Matrix(jW,iW,2) - pE(iE) + YTransfer(iY) - ...
                        pFactor * Egrid(iE) * Wgrid(jW) * Pgrid(jP)^(1+phiP);
    
    if c <= lowC || Wgrid(jW) <= lowW
        v2 = -1e9;
    else        
        V0C1 = 0;
        V0C2 = 0;
        for jY = 1:nY
            V0C1 = V0C1 + TransY(iY,jY) * v1(iE,jA,jW,jY,1);
            V0C2 = V0C2 + TransY(iY,jY)*((1-tWP(jW,jP)) * v1(iE,jA,kP(jW,jP),jY,2) + tWP(jW,jP)*v1(iE,jA,kP(jW,jP)+1,jY,2));
        end
        v2 = (((c-lowC)/scalefactor)^(1-gamma) - 1)/(1-gamma) + UWPart(jW) + ...
            + rho2 * (a*Egrid(iE)*V0C2 + (1-a*Egrid(iE))*V0C1);
    end
end

function [v2,PolE,PolA,PolW,PolP,PolI] = ValueComparePI(iE,iA,iW,iY,iD)
    if v2Mov(iA,iW,iY,iD) > v2Stay(iE,iA,iW,iY,iD)
        v2 = v2Mov(iA,iW,iY,iD);
        PolE = PolicyMovE(iA,iW,iY,iD);
        PolA = PolicyMovA(iA,iW,iY,iD);
        PolW = PolicyMovW(iA,iW,iY,iD);
        PolP = PolicyMovP(iA,iW,iY,iD);
        if PolicyMovE(iA,iW,iY,iD) == iE
            PolI = 0;
        else
            PolI = 1;
        end
    else
        v2 = v2Stay(iE,iA,iW,iY,iD);
        PolE = iE;
        PolA = PolicyStayA(iE,iA,iW,iY,iD);
        PolW = PolicyStayW(iE,iA,iW,iY,iD);
        PolP = PolicyStayP(iE,iA,iW,iY,iD);
        PolI = 0;
    end
end


%% ---------------------------------------------------------------------------------------------------------------------
%   Additional variables
%-----------------------------------------------------------------------------------------------------------------------

function [w,c,Vul] = AdditionalVariables(iE,iA,iW,iY,iD)
    
    jE = PolicyE(iE,iA,iW,iY,iD);
    jA = PolicyA(iE,iA,iW,iY,iD);
    jW = PolicyW(iE,iA,iW,iY,iD);
    jP = PolicyP(iE,iA,iW,iY,iD);
    I = PolicyI(iE,iA,iW,iY,iD);
    
    if Agrid(iA) > 0
        Ahelp = (1+rP)*Agrid(iA);
    else
        Ahelp = (1+rM)*Agrid(iA);
    end
    
    if I == 1
        w  = wMatrix(jW,iW,1);  %Wealth investments
        c = Ygrid(iY)*(1-Deltay*Dgrid(iD)) + Ahelp - Agrid(jA) - w2Matrix(jW,iW,1) - pE(jE) + YTransfer(iY) - pFactor * Egrid(jE) * Wgrid(jW) * Pgrid(jP)^(1+phiP); %Consumption
    else
        w  = wMatrix(jW,iW,2);  %Wealth investments
        c = Ygrid(iY)*(1-Deltay*Dgrid(iD)) + Ahelp - Agrid(jA) - w2Matrix(jW,iW,2) - pE(jE) + YTransfer(iY)- pFactor * Egrid(jE) * Wgrid(jW) * Pgrid(jP)^(1+phiP); %Consumption
    end
    
    ExpU1 = 0;
    ExpU2 = 0;
    for jY = 1:nY
        ExpU1 = ExpU1 + TransY(iY,jY) * v0(jE,jA,jW,jY,1);
        ExpU2 = ExpU2 + TransY(iY,jY) * ((1-tWP(jW,jP)) * v0(jE,jA,kP(jW,jP),jY,2) + tWP(jW,jP)*v0(jE,jA,kP(jW,jP)+1,jY,2));
    end
    Vul = (ExpU1 - ExpU2)/(ExpU1-minV0);
end

function [w,c,Vul] = AdditionalVariablesMov(iA,iW,iY,iD)
    
    jE = PolicyMovE(iA,iW,iY,iD);
    jA = PolicyMovA(iA,iW,iY,iD);
    jW = PolicyMovW(iA,iW,iY,iD);
    jP = PolicyMovP(iA,iW,iY,iD);
    
    if Agrid(iA) > 0
        Ahelp = (1+rP)*Agrid(iA);
    else
        Ahelp = (1+rM)*Agrid(iA);
    end
    
    w  = wMatrix(jW,iW,1);  %Wealth investments
    c = Ygrid(iY)*(1-Deltay*Dgrid(iD)) + Ahelp - Agrid(jA) - w2Matrix(jW,iW,1) - pE(jE) + YTransfer(iY) - pFactor * Egrid(jE) * Wgrid(jW) * Pgrid(jP)^(1+phiP); %Consumption
    
    ExpU1 = 0;
    ExpU2 = 0;
    for jY = 1:nY
        ExpU1 = ExpU1 + TransY(iY,jY) * v0(jE,jA,jW,jY,1);
        ExpU2 = ExpU2 + TransY(iY,jY) * ((1-tWP(jW,jP)) * v0(jE,jA,kP(jW,jP),jY,2) + tWP(jW,jP)*v0(jE,jA,kP(jW,jP)+1,jY,2));
    end
    Vul = (ExpU1 - ExpU2)/(ExpU1-minV0);
end

function [w,c,Vul] = AdditionalVariablesStay(iE,iA,iW,iY,iD)
    
    jE = iE;
    jA = PolicyStayA(iE,iA,iW,iY,iD);
    jW = PolicyStayW(iE,iA,iW,iY,iD);
    jP = PolicyStayP(iE,iA,iW,iY,iD);
    
    if Agrid(iA) > 0
        Ahelp = (1+rP)*Agrid(iA);
    else
        Ahelp = (1+rM)*Agrid(iA);
    end
    
    w  = wMatrix(jW,iW,2);  %Wealth investments
    c = Ygrid(iY)*(1-Deltay*Dgrid(iD)) + Ahelp - Agrid(jA) - w2Matrix(jW,iW,2) - pE(jE) + YTransfer(iY) - pFactor * Egrid(jE) * Wgrid(jW) * Pgrid(jP)^(1+phiP); %Consumption
    
    ExpU1 = 0;
    ExpU2 = 0;
    for jY = 1:nY
        ExpU1 = ExpU1 + TransY(iY,jY) * v0(jE,jA,jW,jY,1);
        ExpU2 = ExpU2 + TransY(iY,jY) * ((1-tWP(jW,jP)) * v0(jE,jA,kP(jW,jP),jY,2) + tWP(jW,jP)*v0(jE,jA,kP(jW,jP)+1,jY,2));
    end
    Vul = (ExpU1 - ExpU2)/(ExpU1-minV0);
end

%% ---------------------------------------------------------------------------------------------------------------------
%   Monotonicity checks
%-----------------------------------------------------------------------------------------------------------------------


function [MonotonicityA,MonotonicityW,MonotonicityY,MonotonicityD] = TestMonotonicityV0(iE,iA,iW,iY,iD)
    MonotonicityA = 1;
    MonotonicityW = 1; 
    MonotonicityY = 1;
    for jA = 1:nA-1
        if v0(iE,jA+1,iW,iY,iD) - v0(iE,jA,iW,iY,iD) < 0
            MonotonicityA = 0;
            break
        end
    end
    for jW = 1:nW-1
        if v0(iE,iA,jW+1,iY,iD) - v0(iE,iA,jW,iY,iD) < 0
            MonotonicityW = 0;
            break
        end
    end
    for jY = 1:nY-1
        if v0(iE,iA,iW,jY+1,iD) - v0(iE,iA,iW,jY,iD) < 0
            MonotonicityY = 0;
            break
        end
    end
    if v0(iE,iA,iW,iY,1) < v0(iE,iA,iW,iY,2)
        MonotonicityD = 0;
    else
        MonotonicityD = 1;
    end
end

function [MonotonicityA,MonotonicityW,MonotonicityY,MonotonicityD] = TestMonotonicityV1(iE,iA,iW,iY,iD)
    MonotonicityA = 1;
    MonotonicityW = 1;
    MonotonicityY = 1;
    for jA = 1:nA-1
        if v1(iE,jA+1,iW,iY,iD) - v1(iE,jA,iW,iY,iD) < 0
            MonotonicityA = 0;
            break
        end
    end
    for jW = 1:nW-1
        if v1(iE,iA,jW+1,iY,iD) - v1(iE,iA,jW,iY,iD) < 0
            MonotonicityW = 0;
            break
        end
    end
    for jY = 1:nY-1
        if v1(iE,iA,iW,jY+1,iD) - v1(iE,iA,iW,jY,iD) < 0
            MonotonicityY = 0;
            break
        end
    end
    if v1(iE,iA,iW,iY,1) < v1(iE,iA,iW,iY,2)
        MonotonicityD = 0;
    else
        MonotonicityD = 1;
    end
end

function [MonotonicityA,MonotonicityW,MonotonicityY,MonotonicityD] = TestMonotonicityV2(iE,iA,iW,iY,iD)
    MonotonicityA = 1;
    MonotonicityW = 1;
    MonotonicityY = 1;
    for jA = 1:nA-1
        if v2(iE,jA+1,iW,iY,iD) - v2(iE,jA,iW,iY,iD) < 0
            MonotonicityA = 0;
            break
        end
    end
    for jW = 1:nW-1
        if v2(iE,iA,jW+1,iY,iD) - v2(iE,iA,jW,iY,iD) < 0
            MonotonicityW = 0;
            break
        end
    end
    for jY = 1:nY-1
        if v2(iE,iA,iW,jY+1,iD) - v2(iE,iA,iW,jY,iD) < 0
            MonotonicityY = 0;
            break
        end
    end
    if v2(iE,iA,iW,iY,1) < v2(iE,iA,iW,iY,2)
        MonotonicityD = 0;
    else
        MonotonicityD = 1;
    end
end

function [ConcavityA,ConcavityW] = TestConcavityV0(iE,iA,iW,iY,iD)
    ConcavityA = 1;
    ConcavityW = 1;   
    for jA = 1:nA-2
        if (v0(iE,jA+1,iW,iY,iD) - v0(iE,jA,iW,iY,iD))/(Agrid(jA+1)-Agrid(jA)) < (v0(iE,jA+2,iW,iY,iD) - v0(iE,jA+1,iW,iY,iD))/(Agrid(jA+2)-Agrid(jA+1))
            ConcavityA = 0;
            break
        end
    end
    for jW = 1:nW-1
        if(v0(iE,iA,jW+1,iY,iD) - v0(iE,iA,jW,iY,iD))/(Wgrid(jW+1)-Wgrid(jW)) < (v0(iE,iA,jW+2,iY,iD) - v0(iE,iA,jW+1,iY,iD))/(Wgrid(jW+2)-Wgrid(jW+1))
            ConcavityW = 0;
            break
        end
    end
end

function [ConcavityA,ConcavityW] = TestConcavityV2(iE,iA,iW,iY,iD)
    ConcavityA = 1;
    ConcavityW = 1;   
    for jA = 1:nA-2
        if (v2(iE,jA+1,iW,iY,iD) - v2(iE,jA,iW,iY,iD))/(Agrid(jA+1)-Agrid(jA)) < (v2(iE,jA+2,iW,iY,iD) - v2(iE,jA+1,iW,iY,iD))/(Agrid(jA+2)-Agrid(jA+1))
            ConcavityA = 0;
            break
        end
    end
    for jW = 1:nW-2
        if(v2(iE,iA,jW+1,iY,iD) - v2(iE,iA,jW,iY,iD))/(Wgrid(jW+1)-Wgrid(jW)) < (v2(iE,iA,jW+2,iY,iD) - v2(iE,iA,jW+1,iY,iD))/(Wgrid(jW+2)-Wgrid(jW+1))
            ConcavityW = 0;
            break
        end
    end
end


%% ---------------------------------------------------------------------------------------------------------------------
%   Initialise Final result arrays
%-----------------------------------------------------------------------------------------------------------------------

Policy.iE.Opt = int16(zeros(nH0,nrho,na,nphiP,nE,nA,nW,nY,nD));
Policy.iA.Opt = int16(zeros(nH0,nrho,na,nphiP,nE,nA,nW,nY,nD));
Policy.iW.Opt = int16(zeros(nH0,nrho,na,nphiP,nE,nA,nW,nY,nD));
Policy.iP.Opt = int16(zeros(nH0,nrho,na,nphiP,nE,nA,nW,nY,nD));

Policy.E.Dis = zeros(nH0,nrho,na,nphiP,nE,nA,nW,nY,nD);
Policy.A.Dis = zeros(nH0,nrho,na,nphiP,nE,nA,nW,nY,nD);
Policy.W.Dis = zeros(nH0,nrho,na,nphiP,nE,nA,nW,nY,nD);
Policy.P.Dis = zeros(nH0,nrho,na,nphiP,nE,nA,nW,nY,nD);
Policy.I.Dis = int16(zeros(nH0,nrho,na,nphiP,nE,nA,nW,nY,nD));

Policy.iE.Stay = int16(zeros(nH0,nrho,na,nphiP,nE,nA,nW,nY,nD));
Policy.iA.Stay = int16(zeros(nH0,nrho,na,nphiP,nE,nA,nW,nY,nD));
Policy.iW.Stay = int16(zeros(nH0,nrho,na,nphiP,nE,nA,nW,nY,nD));
Policy.iP.Stay = int16(zeros(nH0,nrho,na,nphiP,nE,nA,nW,nY,nD));
Policy.iE.Mov = int16(zeros(nH0,nrho,na,nphiP,nA,nW,nY,nD));
Policy.iA.Mov = int16(zeros(nH0,nrho,na,nphiP,nA,nW,nY,nD));
Policy.iW.Mov = int16(zeros(nH0,nrho,na,nphiP,nA,nW,nY,nD));
Policy.iP.Mov = int16(zeros(nH0,nrho,na,nphiP,nA,nW,nY,nD));

Policy.E.Stay = zeros(nH0,nrho,na,nphiP,nE,nA,nW,nY,nD);
Policy.A.Stay = zeros(nH0,nrho,na,nphiP,nE,nA,nW,nY,nD);
Policy.W.Stay = zeros(nH0,nrho,na,nphiP,nE,nA,nW,nY,nD);
Policy.P.Stay = zeros(nH0,nrho,na,nphiP,nE,nA,nW,nY,nD);
Policy.c.Stay = zeros(nH0,nrho,na,nphiP,nE,nA,nW,nY,nD);
Policy.w.Stay = zeros(nH0,nrho,na,nphiP,nE,nA,nW,nY,nD);
Policy.Vul.Stay = zeros(nH0,nrho,na,nphiP,nE,nA,nW,nY,nD);

Policy.E.Mov = zeros(nH0,nrho,na,nphiP,nA,nW,nY,nD);
Policy.A.Mov = zeros(nH0,nrho,na,nphiP,nA,nW,nY,nD);
Policy.W.Mov = zeros(nH0,nrho,na,nphiP,nA,nW,nY,nD);
Policy.P.Mov = zeros(nH0,nrho,na,nphiP,nA,nW,nY,nD);
Policy.c.Mov = zeros(nH0,nrho,na,nphiP,nA,nW,nY,nD);
Policy.w.Mov = zeros(nH0,nrho,na,nphiP,nA,nW,nY,nD);
Policy.Vul.Mov = zeros(nH0,nrho,na,nphiP,nA,nW,nY,nD);

ValueFunction.v0 = zeros(nH0,nrho,na,nphiP,nE,nA,nW,nY,nD);
ValueFunction.v0Dis = zeros(nH0,nrho,na,nphiP,nE,nA,nW,nY,nD);
ValueFunction.Stay.Dis = zeros(nH0,nrho,na,nphiP,nE,nA,nW,nY,nD);
ValueFunction.Mov.Dis = zeros(nH0,nrho,na,nphiP,nA,nW,nY,nD);


%% ---------------------------------------------------------------------------------------------------------------------
%   Actual Algorithm
%-----------------------------------------------------------------------------------------------------------------------

%v0 = 0 * log(Agrid(Agrid5)-Agrid(1)+1e-6) .* log(Wgrid(Wgrid5)-Wgrid(1)+1e-6);

PolicyIold = zeros(nE,nA,nW,nY,nD);
PolicyEold = zeros(nE,nA,nW,nY,nD);
PolicyAold = zeros(nE,nA,nW,nY,nD);
PolicyWold = zeros(nE,nA,nW,nY,nD);
PolicyPold = zeros(nE,nA,nW,nY,nD);

ResetInits = 1;
minV0 = -100;
for iH0 = nH0:(-1):1
    for irho = 1:nrho
        for ia = 1:na
            for iphiP = 1:nphiP
                if isempty(LoadResultsVersion)
                    highA = max(Agrid(end),Para.highA(iH0));
                    highW = max(Wgrid(end),Para.highW(iH0));
                else
                    AgridOld = squeeze(GridOld.A(iH0,irho,ia,iphiP,:))';
                    WgridOld = squeeze(GridOld.W(iH0,irho,ia,iphiP,:))';
                    YgridOld = squeeze(GridOld.Y(iH0,:));
                    highA = AgridOld(end);
                    highW = WgridOld(end);
                end

                if pE(nE) + lowW + lowC > Ygrid_H0(iH0,1)*(1-Deltay) + YTransfer_H0(iH0,1)
                    lowA = 0;
                else
                    lowA = -(Ygrid_H0(iH0,1)*(1-Deltay) - YTransfer_H0(iH0,1) - pE(nE) - lowW - lowC)/rM + 1e-2;
                end
                [Agrid,Wgrid] = CreateAWGrids(lowA,highA,nA,highW,nW,Para);
                Grid.A(iH0,irho,ia,iphiP,:) = Agrid;
                Grid.W(iH0,irho,ia,iphiP,:) = Wgrid;

                [Wgridtemp,W2gridtemp,Igridtemp] = ndgrid(gpuArray(1:nW),gpuArray(1:nW),gpuArray(1:2));
                [WgridtempV2,Pgridtemp] = ndgrid(gpuArray(1:nW),gpuArray(1:nP));

                [wMatrix,w2Matrix] = arrayfun(@PreCalculations,Wgridtemp,W2gridtemp,Igridtemp);
                UWPart = arrayfun(@UW,gpuArray(1:nW));
                [kP,tWP] = arrayfun(@FindPindex,WgridtempV2,Pgridtemp);

                %-----------------------------------------------------------------------------------------------------------
                %   Start discrete Search

                fprintf("Start combination iH0 = %d | irho = %d | ia = %d | iphiP = %d \n",iH0,irho,ia,iphiP)
                TotalParameterTime = tic;

                Diff = 1;
                iter = 1;
                stophighStates = 0;
                STOP = 0;

                rho = rhogrid(irho);
                rho2 = 1/(1+rho);
                a = agrid(ia);
                phiP = phiPgrid(iphiP);
                Ygrid = squeeze(Ygrid_H0(iH0,:));
                TransY = squeeze(TransY_H0(iH0,:,:));
                YTransfer = squeeze(YTransfer_H0(iH0,:));

                %-----------------------------------------------------------------------------------------------------------
                %   Load Initial variables

                if not(isempty(LoadResultsVersion))
                    for iD = 1:2
                        [Egrid4Old,Agrid4Old,Wgrid4Old,Ygrid4Old] = ndgrid(EgridOld,AgridOld,WgridOld,YgridOld);
                        [Egrid4New,Agrid4New,Wgrid4New,Ygrid4New] = ndgrid(Egrid,Agrid,Wgrid,Ygrid);
                        v0(:,:,:,:,iD) = interpn(Egrid4Old,Agrid4Old,Wgrid4Old,Ygrid4Old,...
                                                    squeeze(ValueFunctionOld.v0(iH0,irho,ia,iphiP,:,:,:,:,iD)),...
                                                    Egrid4New,Agrid4New,Wgrid4New,Ygrid4New);
                        PolicyE(:,:,:,:,iD) = interpn(Egrid4Old,Agrid4Old,Wgrid4Old,Ygrid4Old,...
                                                        EgridOld(squeeze(PolicyOld.iE.Opt(iH0,irho,ia,iphiP,:,:,:,:,iD))),...
                                                        Egrid4New,Agrid4New,Wgrid4New,Ygrid4New);
                        PolicyI(:,:,:,:,iD) = interpn(Egrid4Old,Agrid4Old,Wgrid4Old,Ygrid4Old,...
                                                        double(squeeze(PolicyOld.I.Dis(iH0,irho,ia,iphiP,:,:,:,:,iD))),...
                                                        Egrid4New,Agrid4New,Wgrid4New,Ygrid4New);
                                                   
                        PolicyA(:,:,:,:,iD) = interpn(Egrid4Old,Agrid4Old,Wgrid4Old,Ygrid4Old,...
                                                        AgridOld(squeeze(PolicyOld.iA.Opt(iH0,irho,ia,iphiP,:,:,:,:,iD))),...
                                                        Egrid4New,Agrid4New,Wgrid4New,Ygrid4New);

                        PolicyW(:,:,:,:,iD) = interpn(Egrid4Old,Agrid4Old,Wgrid4Old,Ygrid4Old,...
                                                        WgridOld(squeeze(PolicyOld.iW.Opt(iH0,irho,ia,iphiP,:,:,:,:,iD))),...
                                                        Egrid4New,Agrid4New,Wgrid4New,Ygrid4New);

                        PolicyP(:,:,:,:,iD) = interpn(Egrid4Old,Agrid4Old,Wgrid4Old,Ygrid4Old,...
                                                        PgridOld(squeeze(PolicyOld.iP.Opt(iH0,irho,ia,iphiP,:,:,:,:,iD))),...
                                                        Egrid4New,Agrid4New,Wgrid4New,Ygrid4New);

                        PolicyStayA(:,:,:,:,iD) = interpn(Egrid4Old,Agrid4Old,Wgrid4Old,Ygrid4Old,...
                                                        AgridOld(squeeze(PolicyOld.iA.Stay(iH0,irho,ia,iphiP,:,:,:,:,iD))),...
                                                        Egrid4New,Agrid4New,Wgrid4New,Ygrid4New);

                        PolicyStayW(:,:,:,:,iD) = interpn(Egrid4Old,Agrid4Old,Wgrid4Old,Ygrid4Old,...
                                                        WgridOld(squeeze(PolicyOld.iW.Stay(iH0,irho,ia,iphiP,:,:,:,:,iD))),...
                                                        Egrid4New,Agrid4New,Wgrid4New,Ygrid4New);

                        PolicyStayP(:,:,:,:,iD) = interpn(Egrid4Old,Agrid4Old,Wgrid4Old,Ygrid4Old,...
                                                        PgridOld(squeeze(PolicyOld.iP.Stay(iH0,irho,ia,iphiP,:,:,:,:,iD))),...
                                                        Egrid4New,Agrid4New,Wgrid4New,Ygrid4New);

                        if AgridOld(1) > Agrid(1)
                            iA = find(Agrid > AgridOld(1),1);
                            for iE = 1:nE
                                for iW = 1:nW
                                    for iY = 1:nY
                                        v0(iE,1:iA-1,iW,iY,iD) = interp1(Agrid(iA:nA),squeeze(v0(iE,iA:nA,iW,iY,iD)),Agrid(1:iA-1),'linear','extrap');
                                        PolicyE(iE,1:iA-1,iW,iY,iD) = interp1(Agrid(iA:nA),squeeze(PolicyE(iE,iA:nA,iW,iY,iD)),Agrid(1:iA-1),'linear','extrap');
                                        PolicyI(iE,1:iA-1,iW,iY,iD) = interp1(Agrid(iA:nA),squeeze(PolicyI(iE,iA:nA,iW,iY,iD)),Agrid(1:iA-1),'linear','extrap');
                                        PolicyA(iE,1:iA-1,iW,iY,iD) = interp1(Agrid(iA:nA),squeeze(PolicyA(iE,iA:nA,iW,iY,iD)),Agrid(1:iA-1),'linear','extrap');
                                        PolicyW(iE,1:iA-1,iW,iY,iD) = interp1(Agrid(iA:nA),squeeze(PolicyW(iE,iA:nA,iW,iY,iD)),Agrid(1:iA-1),'linear','extrap');
                                        PolicyP(iE,1:iA-1,iW,iY,iD) = interp1(Agrid(iA:nA),squeeze(PolicyP(iE,iA:nA,iW,iY,iD)),Agrid(1:iA-1),'linear','extrap');
                                        PolicyStayA(iE,1:iA-1,iW,iY,iD) = interp1(Agrid(iA:nA),squeeze(PolicyStayA(iE,iA:nA,iW,iY,iD)),Agrid(1:iA-1),'linear','extrap');
                                        PolicyStayW(iE,1:iA-1,iW,iY,iD) = interp1(Agrid(iA:nA),squeeze(PolicyStayW(iE,iA:nA,iW,iY,iD)),Agrid(1:iA-1),'linear','extrap');
                                        PolicyStayP(iE,1:iA-1,iW,iY,iD) = interp1(Agrid(iA:nA),squeeze(PolicyStayP(iE,iA:nA,iW,iY,iD)),Agrid(1:iA-1),'linear','extrap');
                                    end
                                end
                            end
                        end
                        PolicyE(:,:,:,:,iD) = interp1(Egrid,1:nE,PolicyE(:,:,:,:,iD),'nearest','extrap');
                        PolicyI(:,:,:,:,iD) = round(PolicyI(:,:,:,:,iD));
                        PolicyA(:,:,:,:,iD) = interp1(Agrid,1:nA,PolicyA(:,:,:,:,iD),'nearest','extrap');
                        PolicyW(:,:,:,:,iD) = round(interp1(Wgrid,1:nW,PolicyW(:,:,:,:,iD),'nearest','extrap'));
                        PolicyP(:,:,:,:,iD) = round(interp1(Pgrid,1:nP,PolicyP(:,:,:,:,iD),'nearest','extrap'));
                        PolicyStayA(:,:,:,:,iD) = round(interp1(Agrid,1:nA,PolicyStayA(:,:,:,:,iD),'nearest','extrap'));
                        PolicyStayW(:,:,:,:,iD) = round(interp1(Wgrid,1:nW,PolicyStayW(:,:,:,:,iD),'nearest','extrap'));
                        PolicyStayP(:,:,:,:,iD) = round(interp1(Pgrid,1:nP,PolicyStayP(:,:,:,:,iD),'nearest','extrap'));
                    end
                elseif ResetInits == 1
                    v0 = zeros(nE,nA,nW,nY,nD);
                    PolicyIold = zeros(nE,nA,nW,nY,nD);
                    PolicyEold = zeros(nE,nA,nW,nY,nD);
                    PolicyAold = zeros(nE,nA,nW,nY,nD);
                    PolicyWold = zeros(nE,nA,nW,nY,nD);
                    PolicyPold = zeros(nE,nA,nW,nY,nD);
                end


                %-----------------------------------------------------------------------------------------------------------
                %   Discrete Grid-Search
                NothingChanged = 1;
                if isempty(LoadResultsVersion)
                    NothingChanged = 0;
                elseif numel(fieldnames(ParaPol)) == 0
                    if ~isequal(EgridOld,Egrid) 
                        NothingChanged = 0;
                    end
                    if ~isequal(AgridOld,Agrid)
                        NothingChanged = 0;
                    end
                    if ~isequal(WgridOld,Wgrid)
                        NothingChanged = 0;
                    end
                    if ~isequal(YgridOld,Ygrid)
                        NothingChanged = 0;
                    end
                else 
                    if isfield(fieldnames(ParaPol),'Ymin')
                        if any(YTransfer > 0.0)
                            NothingChanged = 0;
                        end
                    else
                        NothingChanged = 0;
                    end
                end
                  
                Para.NothingChanged(iH0,irho,ia,iphiP) = NothingChanged;
                if NothingChanged == 1
                    fprintf("Nothing changed compared to loaded results!\n")
                else
                    TotalGridSearchTime = tic;
                    while STOP == 0

                        TotalIterationTime = tic;

                        MovSearchTime = tic;
                        [v1MovHelp,PolAMovHelp,PolWMovHelp,PolPMovHelp] = arrayfun(@OptiSearchMov,Egrid5,Agrid5,Wgrid5,Ygrid5,Dgrid5);
                        [v1Mov,PolicyMovE,PolicyMovA,PolicyMovW,PolicyMovP] = arrayfun(@OptiSearchMovCompare,Agrid4,Wgrid4,Ygrid4,Dgrid4);
                        fprintf("Time for MovSearch  = %g seconds \n",toc(MovSearchTime))

                        StaySearchTime = tic;
                        [v1Stay,PolicyStayA,PolicyStayW,PolicyStayP] = arrayfun(@OptiSearchStay,Egrid5,Agrid5,Wgrid5,Ygrid5,Dgrid5);
                        fprintf("Time for StaySearch = %g seconds \n",toc(StaySearchTime))

                        [v1,PolicyE,PolicyA,PolicyW,PolicyP,PolicyI] = arrayfun(@ValueCompare,Egrid5,Agrid5,Wgrid5,Ygrid5,Dgrid5);

                        [MonotonicityA,MonotonicityW,MonotonicityY,MonotonicityD] = arrayfun(@TestMonotonicityV1,Egrid5,Agrid5,Wgrid5,Ygrid5,Dgrid5);
                        fprintf("v1:        | Mon A = %d | Mon W = %d | Mon Y = %d | Mon D = %d \n",...
                                all(MonotonicityA,'all'),all(MonotonicityW,'all'),all(MonotonicityY,'all'),all(MonotonicityD,'all'))

                        %-----------------------------------------------------------------------------------------------------------
                        %   Policy Iteration
                        if iter > 2 || ~isempty(LoadResultsVersion)
                            iter2 = 0;
                            Diff2 = 1;
                            while iter2 < 50 && Diff2 > 1e-7
                                v2Mov = arrayfun(@PolicyIterationMov,Agrid4,Wgrid4,Ygrid4,Dgrid4);
                                v2Stay = arrayfun(@PolicyIterationStay,Egrid5,Agrid5,Wgrid5,Ygrid5,Dgrid5);
                                [v2,PolicyE,PolicyA,PolicyW,PolicyP,PolicyI] = arrayfun(@ValueComparePI,Egrid5,Agrid5,Wgrid5,Ygrid5,Dgrid5);

                                Diff2 = max(abs(v2 - v1),[],'all');
                                [MonotonicityA,MonotonicityW,MonotonicityY,MonotonicityD] = arrayfun(@TestMonotonicityV2,Egrid5,Agrid5,Wgrid5,Ygrid5,Dgrid5);
                                %[ConcavityA,ConcavityW] = arrayfun(@TestConcavityV2,Egrid5,Agrid5,Wgrid5,Ygrid5,Dgrid5);

                                if all(MonotonicityA,'all') && all(MonotonicityW,'all') && all(MonotonicityY,'all') && all(MonotonicityD,'all')
                                    v1 = v2;
                                    iter2 = iter2 + 1;
                                    if iter2 == 50 || Diff2 < 1e-7
                                        fprintf("iter2 = %d | Mon A = %d | Mon W = %d | Mon Y = %d | Mon D = %d | Diff2 = %g\n",...
                                        iter2,all(MonotonicityA,'all'),all(MonotonicityW,'all'),all(MonotonicityY,'all'),all(MonotonicityD,'all'),Diff2)
                                    end
                                else
                                    fprintf("iter2 = %d | Mon A = %d | Mon W = %d | Mon Y = %d | Mon D = %d \n",...
                                        iter2,all(MonotonicityA,'all'),all(MonotonicityW,'all'),all(MonotonicityY,'all'),all(MonotonicityD,'all'))
                                    break
                                end
                            end
                        end
                        %-----------------------------------------------------------------------------------------------------------
                        %   Compare Improvement
                        Diff = max(abs(v1 - v0),[],'all');
                        fprintf("ITERATION %d | Total Time = %g seconds | Diff = %g | ",iter,toc(TotalIterationTime),Diff)
                        fprintf("Combination iH0 = %d | irho = %d | ia = %d | iphiP = %d \n",iH0,irho,ia,iphiP)

                        v0 = v1;
                        iter = iter + 1;

                        %-----------------------------------------------------------------------------------------------------------
                        %   Conduct total grid search if convergence is 'completed'
                        if (all(PolicyI == PolicyIold,'all') && all(PolicyE == PolicyEold,'all') && ...
                                all(PolicyA == PolicyAold,'all') && all(PolicyW == PolicyWold,'all') && ...
                                all(PolicyP == PolicyPold,'all')) || Diff < 1e-5 %|| mod(iter,5) == 0

                            v1MovComp = v1Mov;
                            v1StayComp = v1Stay;
                            MovSearchTime = tic;
                            [v1MovHelp,PolAMovHelp,PolWMovHelp,PolPMovHelp] = arrayfun(@OptiSearchMovAllReduced,Egrid5,Agrid5,Wgrid5,Ygrid5,Dgrid5);
                            [v1Mov,PolicyMovE,PolicyMovA,PolicyMovW,PolicyMovP] = arrayfun(@OptiSearchMovCompare,Agrid4,Wgrid4,Ygrid4,Dgrid4);
                            fprintf("Time for MovSearchALL  = %g seconds \n",toc(MovSearchTime))

                            StaySearchTime = tic;
                            [v1Stay,PolicyStayA,PolicyStayW,PolicyStayP] = arrayfun(@OptiSearchStayAllReduced,Egrid5,Agrid5,Wgrid5,Ygrid5,Dgrid5);
                            fprintf("Time for StaySearchALL = %g seconds \n",toc(StaySearchTime))

                            [v1,PolicyE,PolicyA,PolicyW,PolicyP,PolicyI] = arrayfun(@ValueCompare,Egrid5,Agrid5,Wgrid5,Ygrid5,Dgrid5);

                            fprintf("Difference Mov  = %g | # Difference Mov  = %g \n",...
                                max(v1Mov - v1MovComp,[],'all'),length(find(abs(v1Mov - v1MovComp)>1e-6)))
                            fprintf("Difference Stay = %g | # Difference Stay = %g \n",...
                                max(v1Stay - v1StayComp,[],'all'),length(find(abs(v1Stay - v1StayComp)>1e-6)))
                            fprintf("Difference Total = %g | # Difference Total = %g \n" ,...
                                max(v1 - v0,[],'all'),length(find(abs(v1 - v0)>1e-6)))

                            if length(find(abs(v1 - v0)>1e-6)) < 0.001*nE*nA*nW*nY*nD && max(abs(v1 - v0),[],'all') < 1e-4
                                Diff = 1e-7;                        
                            elseif max(v1Mov - v1MovComp,[],'all') > 1e-6 || (max(v1Stay - v1StayComp,[],'all') > 1e-6)
                                iter2 = 0;
                                Diff2 = 1;
                                while iter2 < 20 && Diff2 > 1e-7 %&& mod(iter,4) == 0
                                    v2Mov = arrayfun(@PolicyIterationMov,Agrid4,Wgrid4,Ygrid4,Dgrid4);
                                    v2Stay = arrayfun(@PolicyIterationStay,Egrid5,Agrid5,Wgrid5,Ygrid5,Dgrid5);
                                    [v2,PolicyE,PolicyA,PolicyW,PolicyP,PolicyI] = arrayfun(@ValueComparePI,Egrid5,Agrid5,Wgrid5,Ygrid5,Dgrid5);
                                    Diff2 = max(abs(v2 - v1),[],'all');
                                    [MonotonicityA,MonotonicityW,MonotonicityY,MonotonicityD] = arrayfun(@TestMonotonicityV2,Egrid5,Agrid5,Wgrid5,Ygrid5,Dgrid5);

                                    if all(MonotonicityA,'all') && all(MonotonicityW,'all') && all(MonotonicityY,'all') && all(MonotonicityD,'all')
                                        v1 = v2;
                                        iter2 = iter2 + 1;
                                        if iter2 == 20
                                            fprintf("iter2 = %d | Mon A = %d | Mon W = %d | Mon Y = %d | Mon D = %d \n",...
                                            iter2,all(MonotonicityA,'all'),all(MonotonicityW,'all'),all(MonotonicityY,'all'),all(MonotonicityD,'all'))
                                            fprintf("-----------------\n")
                                        end
                                    else
                                        fprintf("iter2 = %d | Mon A = %d | Mon W = %d | Mon Y = %d | Mon D = %d \n",...
                                            iter2,all(MonotonicityA,'all'),all(MonotonicityW,'all'),all(MonotonicityY,'all'),all(MonotonicityD,'all'))
                                        fprintf("-----------------\n")
                                        break
                                    end
                                end
                                Diff = max(abs(v1 - v0),[],'all');
                                v0 = v1;
                            end
                        end

                        PolicyEold = PolicyE;
                        PolicyAold = PolicyA;
                        PolicyWold = PolicyW;
                        PolicyPold = PolicyP;
                        PolicyIold = PolicyI;

                        %-----------------------------------------------------------------------------------------------------------
                        %   Check if range of A or W has to be extended
                        if mod(iter,7) == 0 || Diff < 1e-6
                            stophighW = ones(nE,nY,nD);
                            stophighA = ones(nE,nY,nD);

                            MAXiWup = 1;
                            MAXiAup = 1;
                            for iE = 1:nE
                                for iY = 1:nY
                                    for iD = 1:2
                                        if all(PolicyW(iE,:,nW,iY,iD) < nW,'all') && all(PolicyA(iE,nA,:,iY,iD) < nA,'all')
                                            stophighW(iE,iY,iD) = 1;
                                            stophighA(iE,iY,iD) = 1;
                                        else
                                            iWup = nW;
                                            while all(PolicyW(iE,:,iWup,iY,iD) < iWup,'all')
                                                iWup = iWup - 1;
                                                if iWup == 0
                                                    break
                                                end
                                            end

                                            if iWup == nW
                                                stophighW(iE,iY,iD) = 0;
                                            else
                                                iWup = iWup + 1;
                                            end

                                            iAup = nA;
                                            while all(PolicyA(iE,iAup,1:iWup,iY,iD) <= iAup,'all')
                                                iAup = iAup - 1;
                                                if iAup == 0
                                                    break
                                                end
                                            end
                                            iAup = iAup + 1;
                                            if iAup == nA
                                                if any(PolicyA(iE,nA-1,1:iWup,iY,iD) > Agrid5(iE,nA-1,1:iWup,iY,iD),'all')
                                                    stophighA(iE,iY,iD) = 0;
                                                end
                                            end

                                            if iWup == nW
                                                if all(PolicyW(iE,1:iAup,nW,iY,iD) < iWup,'all')
                                                    stophighW(iE,iY,iD) = 1;
                                                end
                                            end

                                            MAXiWup = max(MAXiWup,iWup);
                                            MAXiAup = max(MAXiAup,iAup);
                                        end
                                    end
                                end
                            end

                            stophighStates = min(stophighA(:))*min(stophighW(:));
                            if stophighStates == 1 || (Agrid(end) > 200 || Wgrid(end) > 200)
                                Grid.A(iH0,irho,ia,iphiP,:) = Agrid;
                                Grid.W(iH0,irho,ia,iphiP,:) = Wgrid;
                                stophighStates = 1;
                            end

                            if min(stophighA(:)) == 0 && Agrid(end) <= 200
                                highA = highA*1.5;
                                [Agrid,~] = CreateAWGrids(lowA,highA,nA,highW,nW,Para);
                                Grid.A(iH0,irho,ia,iphiP,:) = Agrid;
                                stophighStates = 0;
                            end

                            if min(stophighW(:)) == 0 && Wgrid(end) <= 200
                                highW = highW*1.5;
                                [~,Wgrid] = CreateAWGrids(lowA,highA,nA,highW,nW,Para);
                                [wMatrix,w2Matrix] = arrayfun(@PreCalculations,Wgridtemp,W2gridtemp,Igridtemp);
                                UWPart = arrayfun(@UW,gpuArray(1:nW));
                                [kP,tWP] = arrayfun(@FindPindex,WgridtempV2,Pgridtemp);
                                stophighStates = 0;
                            end
                            fprintf("stophighStates = %d | highA = %g | highW = %g\n",stophighStates,highA,highW)
                        end

                        if (Diff > 1e-6 && iter < 20) || (iter < 4 && nargin < 5) || stophighStates == 0
                            STOP = 0;
                        else
                            STOP = 1;
                            if iter > 35
                                ResetInits = 1;
                            end
                        end                    
                    end

                    fprintf("TIME FOR TOTAL DISCRETE GRID SEARCH = %g SECONDS \n",toc(TotalGridSearchTime))
                    fprintf("------------------------------------------------------------------------\n")
                    v0Dis = v0;
                end

                %-----------------------------------------------------------------------------------------------------------
                %   Assign results to overall dataframes
                Policy.iE.Opt(iH0,irho,ia,iphiP,:,:,:,:,:) = int16(gather(PolicyE));
                Policy.iA.Opt(iH0,irho,ia,iphiP,:,:,:,:,:) = int16(gather(PolicyA));
                Policy.iW.Opt(iH0,irho,ia,iphiP,:,:,:,:,:) = int16(gather(PolicyW));
                Policy.iP.Opt(iH0,irho,ia,iphiP,:,:,:,:,:) = int16(gather(PolicyP));

                Policy.E.Dis(iH0,irho,ia,iphiP,:,:,:,:,:) = gather(Egrid(Policy.iE.Opt(iH0,irho,ia,iphiP,:,:,:,:,:)));
                Policy.A.Dis(iH0,irho,ia,iphiP,:,:,:,:,:) = gather(Agrid(Policy.iA.Opt(iH0,irho,ia,iphiP,:,:,:,:,:)));
                Policy.W.Dis(iH0,irho,ia,iphiP,:,:,:,:,:) = gather(Wgrid(Policy.iW.Opt(iH0,irho,ia,iphiP,:,:,:,:,:)));
                Policy.P.Dis(iH0,irho,ia,iphiP,:,:,:,:,:) = gather(Pgrid(Policy.iP.Opt(iH0,irho,ia,iphiP,:,:,:,:,:)));
                Policy.I.Dis(iH0,irho,ia,iphiP,:,:,:,:,:) = int16(gather(PolicyI));

                Policy.iE.Stay(iH0,irho,ia,iphiP,:,:,:,:,:) = int64(gather(Egrid5));
                Policy.iA.Stay(iH0,irho,ia,iphiP,:,:,:,:,:) = int64(gather(PolicyStayA));
                Policy.iW.Stay(iH0,irho,ia,iphiP,:,:,:,:,:) = int64(gather(PolicyStayW));
                Policy.iP.Stay(iH0,irho,ia,iphiP,:,:,:,:,:) = int64(gather(PolicyStayP));

                Policy.iE.Mov(iH0,irho,ia,iphiP,:,:,:,:) = int64(gather(PolicyMovE));
                Policy.iA.Mov(iH0,irho,ia,iphiP,:,:,:,:) = int64(gather(PolicyMovA));
                Policy.iW.Mov(iH0,irho,ia,iphiP,:,:,:,:) = int64(gather(PolicyMovW));
                Policy.iP.Mov(iH0,irho,ia,iphiP,:,:,:,:) = int64(gather(PolicyMovP));

                Policy.E.Stay(iH0,irho,ia,iphiP,:,:,:,:,:) = gather(Egrid(Egrid5));
                Policy.A.Stay(iH0,irho,ia,iphiP,:,:,:,:,:) = gather(Agrid(PolicyStayA));
                Policy.W.Stay(iH0,irho,ia,iphiP,:,:,:,:,:) = gather(Wgrid(PolicyStayW));
                Policy.P.Stay(iH0,irho,ia,iphiP,:,:,:,:,:) = gather(Pgrid(PolicyStayP));

                Policy.E.Mov(iH0,irho,ia,iphiP,:,:,:,:) = gather(Egrid(PolicyMovE));
                Policy.A.Mov(iH0,irho,ia,iphiP,:,:,:,:) = gather(Agrid(PolicyMovA));
                Policy.W.Mov(iH0,irho,ia,iphiP,:,:,:,:) = gather(Wgrid(PolicyMovW));
                Policy.P.Mov(iH0,irho,ia,iphiP,:,:,:,:) = gather(Pgrid(PolicyMovP));

                ValueFunction.v0(iH0,irho,ia,iphiP,:,:,:,:,:) = gather(v0);
                ValueFunction.v0Dis(iH0,irho,ia,iphiP,:,:,:,:,:) = gather(v0Dis);
                ValueFunction.Stay.Dis(iH0,irho,ia,iphiP,:,:,:,:,:) = gather(v2Stay);
                ValueFunction.Mov.Dis(iH0,irho,ia,iphiP,:,:,:,:) = gather(v2Mov);

                if min(v0,[],'all') < minV0
                    minV0 = min(v0,[],'all');
                end

                [WealthInv,Consumption,Vul] = arrayfun(@AdditionalVariables,Egrid5,Agrid5,Wgrid5,Ygrid5,Dgrid5);
                [WealthInvMov,ConsumptionMov,VulMov] = arrayfun(@AdditionalVariablesMov,Agrid4,Wgrid4,Ygrid4,Dgrid4);
                [WealthInvStay,ConsumptionStay,VulStay] = arrayfun(@AdditionalVariablesStay,Egrid5,Agrid5,Wgrid5,Ygrid5,Dgrid5);
                Policy.c.Mov(iH0,irho,ia,iphiP,:,:,:,:) = gather(ConsumptionMov);
                Policy.w.Mov(iH0,irho,ia,iphiP,:,:,:,:) = gather(WealthInvMov);
                Policy.Vul.Mov(iH0,irho,ia,iphiP,:,:,:,:) = gather(VulMov);
                Policy.c.Stay(iH0,irho,ia,iphiP,:,:,:,:,:) = gather(ConsumptionStay);
                Policy.w.Stay(iH0,irho,ia,iphiP,:,:,:,:,:) = gather(WealthInvStay);
                Policy.Vul.Stay(iH0,irho,ia,iphiP,:,:,:,:,:) = gather(VulStay);
                Policy.c.Dis(iH0,irho,ia,iphiP,:,:,:,:,:) = gather(Consumption);
                Policy.w.Dis(iH0,irho,ia,iphiP,:,:,:,:,:) = gather(WealthInv);
                Policy.Vul.Dis(iH0,irho,ia,iphiP,:,:,:,:,:) = gather(Vul);

                Policy.NetSavings.Dis(iH0,irho,ia,iphiP,:,:,:,:,:) = Agrid(PolicyA) - Agrid(Agrid5);
                Policy.NetWealthChange.Dis(iH0,irho,ia,iphiP,:,:,:,:,:) = Wgrid(PolicyW) - Wgrid(Wgrid5);

                fprintf("Time for Total Solution = %g seconds \n",toc(TotalParameterTime))
                fprintf("------------------------------------------------------------------------\n")
                fprintf("------------------------------------------------------------------------\n")
            end
        end
        save(strcat(SaveFolder,"\Results_discrete_",Version,".mat"),'ValueFunction','Policy','Grid','Para', '-v7.3')
    end
end

save(strcat(SaveFolder,"\Results_discrete_",Version,".mat"),'ValueFunction','Policy','Grid','Para', '-v7.3')
fprintf("Totaltime = %g seconds\n",toc(totaltime))
end
