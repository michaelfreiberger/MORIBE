
function [ValueFunction,Policy,Grid] = VF_final_continuous(varargin) 

GPU = 1;
Version = 'Test';
SaveFolder = "D:\Users\mfreiber\DisasterRiskModel\Matlab-Simulations_V2.1";

for jj = 1:2:nargin
    if strcmp('GPU', varargin{jj})
        GPU = varargin{jj+1};
    elseif strcmp('Version', varargin{jj})
        Version = varargin{jj+1};
    elseif strcmp('SaveFolder', varargin{jj})
        SaveFolder = varargin{jj+1};
    end
end

gpuDevice(GPU)
totaltime = tic;

DATA = load(strcat(SaveFolder,"\Results_discrete_",Version,".mat"),'ValueFunction','Policy','Grid','Para');

ValueFunctionDis = DATA.ValueFunction;
PolicyDis = DATA.Policy;
Grid = DATA.Grid;
Para = DATA.Para;

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

%% Grids
Egrid = Grid.E;
Agrid = squeeze(Grid.A(1,1,1,1,:))';
Wgrid = squeeze(Grid.W(1,1,1,1,:))';
Dgrid = Grid.D;
Ygrid = Grid.Y(1,:);

Agrid = squeeze(Grid.A(1,1,1,1,:))';
Ygrid = Grid.Y(1,:);
pE = Grid.pE;

nE = length(Egrid);
nA = length(Agrid);
nW = length(Wgrid);
nD = length(Dgrid);
nY = size(Ygrid,2);
nH0 = Para.nH0;
nrho = Para.nrho;
na = Para.na;
nphiP = Para.nphiP;


%% Grid Definitions

[Agrid4,Wgrid4,Ygrid4,Dgrid4] = ndgrid(gpuArray(1:nA),gpuArray(1:nW),gpuArray(1:nY),gpuArray(1:nD) );
%[Agrid4,Wgrid4,Ygrid4,Dgrid4] = ndgrid(1:nA,1:nW,1:nY,1:nD);
[Egrid5,Agrid5,Wgrid5,Ygrid5,Dgrid5] = ndgrid( gpuArray(1:nE), gpuArray(1:nA),gpuArray(1:nW),gpuArray(1:nY),gpuArray(1:nD) );
%[Egrid5,Agrid5,Wgrid5,Ygrid5,Dgrid5] = ndgrid(1:nE,1:nA,1:nW,1:nY,1:nD);

%% Initialize Variables
v0 = zeros(nE,nA,nW,nY,nD);

%% ---------------------------------------------------------------------------------------------------------------------
%   Fine Search
%-----------------------------------------------------------------------------------------------------------------------

FineEpsP2 = 1e-3;
FineEpsP3 = 1e-2;
FineEpsA2 = 1e-3;
FineEpsA3 = 1e-2;
FineEpsW2 = 1e-3;
FineEpsW3 = 1e-2;

ASmooth = 0.05;
wSmooth = 0.05;

function [Value,PolA,PolW,PolP,iterFine,GradNorm] = OptiSearchMovFineGrad(iEstar,iA,iW,iY,iD,StartA,StartW,StartP)

    E0 = Egrid(iEstar);
    jE = iEstar;
    A0 = Agrid(iA);
    W0 = Wgrid(iW);
    Y0 = Ygrid(iY);
    D0 = Dgrid(iD);
    
    GradNorm = 0;
    Anew = 0; Wnew = 0; Pnew = 0;
        
    Astar = StartA;
    jAstar = 0;
    for kA = 2:nA
        if Agrid(kA) > Astar
            jAstar = kA - 1;
            break
        elseif kA == nA
            jAstar = nA-1;
        end
    end
    
    Wstar = StartW;
    jWstar = 0;
    for kW = 2:nW
        if Wgrid(kW) > Wstar
            jWstar = kW - 1;
            break
        elseif kW == nW
            jWstar = nW-1;
        end
    end
    
    Pstar = StartP;
    jPstar = 0;
    for kW = 2:nW
        if Wgrid(kW) > Pstar*Wstar
            jPstar = kW - 1;
            break
        elseif kW == nW
            jPstar = nW-1;
        end
    end
    tAstar = (Astar - Agrid(jAstar))/(Agrid(jAstar+1)-Agrid(jAstar));
    tWstar = (Wstar - Wgrid(jWstar))/(Wgrid(jWstar+1)-Wgrid(jWstar));
    tPstar = (Pstar*Wstar - Wgrid(jPstar))/(Wgrid(jPstar+1)-Wgrid(jPstar));
        
    if A0 >= ASmooth
        TotalIncome = Y0*(1-Deltay*D0) + (1+rP)*A0 - pE(jE) + YTransfer(iY);
    elseif A0 <= -ASmooth
        TotalIncome = Y0*(1-Deltay*D0) + (1+rM)*A0 - pE(jE) + YTransfer(iY);
    else
        r = rM + ((A0/ASmooth)/(1+(A0/ASmooth)^2)+0.5)*(rP-rM);
        TotalIncome = Y0*(1-Deltay*D0) + (1+r)*A0 - pE(jE) + YTransfer(iY);
    end
    
    w = Wstar - (1-delta)*(1-DeltaW)*W0;
    if w <= -wSmooth
        winvest = kappaZ;
    elseif w >= wSmooth
        winvest = 1;
    else
        winvest = kappaZ + ((w/wSmooth)/(1+(w/wSmooth)^2)+0.5)*(1-kappaZ);
    end
    c = TotalIncome - w*winvest - Astar - pFactor * E0 * Wstar * Pstar^(1+phiP);
    
    if c <= lowC
        Astar = Agrid(1);
        jAstar = 0;
        for kA = 2:nA
            if Agrid(kA) > Astar
                jAstar = kA - 1;
                break
            elseif kA == nA
                jAstar = nA-1;
            end
        end
        
        Wstar = lowW + 2e-6;
        jWstar = 0;
        for kW = 2:nW
            if Wgrid(kW) > Wstar
                jWstar = kW - 1;
                break
            elseif kW == nW
                jWstar = nW-1;
            end
        end
        
        Pstar = 0.0;
        jPstar = 0;
        for kW = 2:nW
            if Wgrid(kW) > Pstar*Wstar
                jPstar = kW - 1;
                break
            elseif kW == nW
                jPstar = nW-1;
            end
        end
        tAstar = (Astar - Agrid(jAstar))/(Agrid(jAstar+1)-Agrid(jAstar));
        tWstar = (Wstar - Wgrid(jWstar))/(Wgrid(jWstar+1)-Wgrid(jWstar));
        tPstar = (Pstar*Wstar - Wgrid(jPstar))/(Wgrid(jPstar+1)-Wgrid(jPstar));
        
        w = Wstar - (1-delta)*(1-DeltaW)*W0;
        if w < -0.1
            winvest = kappaZ;
        elseif w > 0.1
            winvest = 1;
        else
            winvest = kappaZ + ((w/0.1)/(1+(w/0.1)^2)+0.5)*(1-kappaZ);
        end
        c = TotalIncome - w*winvest - Astar - pFactor * E0 * Wstar * Pstar^(1+phiP);
        
        if c < lowC + 1e-6
            GlobalStop = 1;
        else
            GlobalStop = 0;
        end 
    else
        GlobalStop = 0;
    end
        
    if c <= lowC || Wstar <= lowW
        ValueStar = -1e9;
    else
        Utility1 = (((c-lowC)/scalefactor)^(1-gamma) - 1)/(1-gamma) + theta * ( ((Wstar-lowW)/scalefactor)^(1-beta) - 1)/(1-beta);
        V0C1 = 0;
        V0C2 = 0;
        for jY = 1:nY
            V0C1 = V0C1 + TransY(iY,jY) * ( (1-tAstar)*((1-tWstar)*v0(jE, jAstar  ,jWstar ,jY,1)   ...
                +  tWstar *v0(jE, jAstar  ,jWstar+1,jY,1))  ...
                + tAstar*((1-tWstar)*v0(jE, jAstar+1,jWstar  ,jY,1)   ...
                +  tWstar *v0(jE, jAstar+1,jWstar+1,jY,1)));
            V0C2 = V0C2 + TransY(iY,jY) * ( (1-tAstar)*((1-tPstar)*v0(jE, jAstar  ,jPstar ,jY,2)   ...
                +  tPstar *v0(jE, jAstar  ,jPstar+1    ,jY,2))  ...
                + tAstar*((1-tPstar)*v0(jE, jAstar+1,jPstar      ,jY,2)   ...
                +  tPstar *v0(jE, jAstar+1,jPstar+1    ,jY,2)));
        end
        ValueStar = Utility1 + rho2 * (a*E0*V0C2 + (1-a*E0)*V0C1);
    end
        
    iterFine = 0;
    stepsize = 1e-3;       
    
    while GlobalStop == 0 && iterFine < 500
        %%%Calculate Gradients
        
        w = Wstar - (1-delta)*(1-DeltaW)*W0;
        if w <= -wSmooth
            winvest = kappaZ;
        elseif w > wSmooth
            winvest = 1;
        else
            winvest = kappaZ + ((w/wSmooth)/(1+(w/wSmooth)^2)+0.5)*(1-kappaZ);
        end
        c = TotalIncome - w*winvest - Astar - pFactor * E0 * Wstar * Pstar^(1+phiP);
        if c > lowC
            dUdc = ((c-lowC)/scalefactor)^(-gamma)/scalefactor;
        else
            dUdc = 1e9;
        end
        if Wstar > lowW
            dUdW = theta * ((Wstar-lowW)/scalefactor)^(-beta)/scalefactor;
        else
            dUdW = 1e9;
        end
        dVdAC1 = 0;
        dVdAC2 = 0;
        for jY = 1:nY
            dVdAC1 = dVdAC1 + TransY(iY,jY) * (((1-tWstar)*v0(jE, jAstar+1,jWstar  ,jY,1) + tWstar*v0(jE, jAstar+1,jWstar+1,jY,1))  ...
                - ((1-tWstar)*v0(jE, jAstar  ,jWstar  ,jY,1) + tWstar *v0(jE, jAstar  ,jWstar+1,jY,1)));
            
            dVdAC2 = dVdAC2 + TransY(iY,jY) * (((1-tPstar)*v0(jE, jAstar+1,jPstar  ,jY,2) + tPstar*v0(jE, jAstar+1,jPstar+1,jY,2))  ...
                - ((1-tPstar)*v0(jE, jAstar  ,jPstar  ,jY,2) + tPstar *v0(jE, jAstar  ,jPstar+1,jY,2)));
        end
        dVdA = ((1-a*E0)*dVdAC1 + a*E0*dVdAC2)/(Agrid(jAstar+1)-Agrid(jAstar));
        GradA = rho2*dVdA - dUdc;
        
        dVdWC1 = 0;
        dVdWC2 = 0;
        for jY = 1:nY
            dVdWC1 = dVdWC1 + TransY(iY,jY) * (((1-tAstar)*v0(jE, jAstar,jWstar+1,jY,1) + tAstar *v0(jE, jAstar+1,jWstar+1,jY,1))  ...
                - ((1-tAstar)*v0(jE, jAstar  ,jWstar ,jY,1) + tAstar *v0(jE, jAstar+1 ,jWstar,jY,1)));
            dVdWC2 = dVdWC2 + TransY(iY,jY) * (((1-tAstar)*v0(jE, jAstar,jPstar+1,jY,2) + tAstar *v0(jE, jAstar+1,jPstar+1,jY,2))  ...
                - ((1-tAstar)*v0(jE, jAstar  ,jPstar ,jY,2) + tAstar *v0(jE, jAstar+1 ,jPstar,jY,2)));
        end
        dVdW = (1-a*E0)*dVdWC1/(Wgrid(jWstar+1)-Wgrid(jWstar)) + Pstar*a*E0*dVdWC2/(Wgrid(jPstar+1)-Wgrid(jPstar));
        GradW = rho2*dVdW - dUdc*(winvest + pFactor*E0*Pstar^(1+phiP)) + dUdW;
        
        dVdP = Wstar*a*E0*dVdWC2/(Wgrid(jPstar+1)-Wgrid(jPstar));
        GradP = rho2*dVdP - dUdc*pFactor*E0*Wstar*(1+phiP)*Pstar^phiP;
        
        %% Scale gradient to 1
        
        if Astar <= Agrid(1) && GradA < 0
            GradA = 0;
        end
        if Wstar <= lowW + 1e-6 && GradW < 0
            GradW = 0;
        end
        if (Pstar <= 0.0 && GradP < 0) || (Pstar >= 1.0 && GradP > 0)
            GradP = 0;
        end
        GradNorm = sqrt(GradA^2+GradW^2+GradP^2);
        GradAnew = GradA/GradNorm;
        GradWnew = GradW/GradNorm;
        GradPnew = GradP/GradNorm;
        
        if GradNorm > 1e-6
            %% Make step in Gradient direction
            LineStepStop = 0;
        
            while LineStepStop == 0
                
                Anew = Astar + stepsize*GradAnew;
                Anew = max(Agrid(1),min(Agrid(nA),Anew));
                Wnew = Wstar + stepsize*GradWnew;
                Wnew = max(lowW+1e-6,min(Wgrid(nW),Wnew));
                Pnew = Pstar + stepsize*GradPnew;
                Pnew = max(0.0,min(1.0,Pnew));
                
                jAnew = 0;
                for kA = 2:nA
                    if Agrid(kA) > Anew
                        jAnew = kA - 1;
                        break
                    elseif kA == nA
                        jAnew = nA-1;
                    end
                end
                
                jWnew = 0;
                for kW = 2:nW
                    if Wgrid(kW) > Wnew
                        jWnew = kW - 1;
                        break
                    elseif kW == nW
                        jWnew = nW-1;
                    end
                end
                
                jPnew = 0;
                for kW = 2:nW
                    if Wgrid(kW) > Pnew*Wnew
                        jPnew = kW - 1;
                        break
                    elseif kW == nW
                        jPnew = nW-1;
                    end
                end
                tAnew = (Anew - Agrid(jAnew))/(Agrid(jAnew+1)-Agrid(jAnew));
                tWnew = (Wnew - Wgrid(jWnew))/(Wgrid(jWnew+1)-Wgrid(jWnew));
                tPnew = (Pnew*Wnew - Wgrid(jPnew))/(Wgrid(jPnew+1)-Wgrid(jPnew));
                
                w = Wnew - (1-delta)*(1-DeltaW)*W0;
                if w <= -wSmooth
                    winvest = kappaZ;
                elseif w > wSmooth
                    winvest = 1;
                else
                    winvest = kappaZ + ((w/wSmooth)/(1+(w/wSmooth)^2)+0.5)*(1-kappaZ);
                end
                c = TotalIncome - w*winvest - Anew - pFactor * E0 * Wnew * Pnew^(1+phiP);
                
                if c <= lowC || Wnew <= lowW
                    ValueStarNew = -1e9;
                else
                    Utility1 = (((c-lowC)/scalefactor)^(1-gamma) - 1)/(1-gamma) + theta * ( ((Wnew-lowW)/scalefactor)^(1-beta) - 1)/(1-beta);
                    V0C1 = 0;
                    V0C2 = 0;
                    for jY = 1:nY
                        V0C1 = V0C1 + TransY(iY,jY) * ( (1-tAnew)*((1-tWnew)*v0(jE, jAnew  ,jWnew ,jY,1)   ...
                            +  tWnew *v0(jE, jAnew  ,jWnew+1,jY,1))  ...
                            + tAnew*((1-tWnew)*v0(jE, jAnew+1,jWnew  ,jY,1)   ...
                            +  tWnew *v0(jE, jAnew+1,jWnew+1,jY,1)));
                        V0C2 = V0C2 + TransY(iY,jY) * ( (1-tAnew)*((1-tPnew)*v0(jE, jAnew  ,jPnew ,jY,2)   ...
                            +  tPnew *v0(jE, jAnew  ,jPnew+1    ,jY,2))  ...
                            + tAnew*((1-tPnew)*v0(jE, jAnew+1,jPnew      ,jY,2)   ...
                            +  tPnew *v0(jE, jAnew+1,jPnew+1    ,jY,2)));
                    end
                    ValueStarNew = Utility1 + rho2 * (a*E0*V0C2 + (1-a*E0)*V0C1);
                end
                
                if ValueStarNew > ValueStar
                    LineStepStop = 1;
                    stepsize = min(1e1,stepsize*1.25);
                    
                    Astar = Anew;
                    Wstar = Wnew;
                    Pstar = Pnew;
                    ValueStar = ValueStarNew;
                    jAstar = jAnew;
                    tAstar = tAnew;
                    jWstar = jWnew;
                    tWstar = tWnew;
                    jPstar = jPnew;
                    tPstar = tPnew;
%                    fprintf(" + | stepsize = %g | GradNorm = %g | ValueStar = %g \n",stepsize,GradNorm,ValueStar)
                elseif stepsize < 1e-9
                    LineStepStop = 1;
                    GlobalStop = 1;
                else
                    stepsize = stepsize*0.5;
 %                   fprintf(" - | stepsize = %g | GradNorm = %g | ValueStar = %g \n",stepsize,GradNorm,ValueStar)
                end
            end
            iterFine = iterFine + 1;
          
        else
            GlobalStop = 1;
        end        
    end
    PolA = Astar;
    PolW = Wstar;
    PolP = Pstar;
    Value = ValueStar;
end

function [Value,PolA,PolW,PolP,iterFine,GradNorm] = OptiSearchMovFineGradSmoothed(iEstar,iA,iW,iY,iD,StartA,StartW,StartP)

    E0 = Egrid(iEstar);
    jE = iEstar;
    A0 = Agrid(iA);
    W0 = Wgrid(iW);
    Y0 = Ygrid(iY);
    D0 = Dgrid(iD);
    
    GradNorm = 0;
    Anew = 0; Wnew = 0; Pnew = 0;
        
    Astar = StartA;
    jAstar = 0;
    for kA = 2:nA
        if Agrid(kA) > Astar
            jAstar = kA - 1;
            break
        elseif kA == nA
            jAstar = nA-1;
        end
    end
    
    Wstar = StartW;
    jWstar = 0;
    for kW = 2:nW
        if Wgrid(kW) > Wstar
            jWstar = kW - 1;
            break
        elseif kW == nW
            jWstar = nW-1;
        end
    end
    
    Pstar = StartP;
    jPstar = 0;
    for kW = 2:nW
        if Wgrid(kW) > Pstar*Wstar
            jPstar = kW - 1;
            break
        elseif kW == nW
            jPstar = nW-1;
        end
    end
    
    tAstar = (Astar - Agrid(jAstar))/(Agrid(jAstar+1)-Agrid(jAstar));
    tWstar = (Wstar - Wgrid(jWstar))/(Wgrid(jWstar+1)-Wgrid(jWstar));
    tPstar = (Pstar*Wstar - Wgrid(jPstar))/(Wgrid(jPstar+1)-Wgrid(jPstar));
        
    if A0 > ASmooth
        TotalIncome = Y0*(1-Deltay*D0) + (1+rP)*A0 - pE(jE) + YTransfer(iY);
    elseif A0 <= -ASmooth
        TotalIncome = Y0*(1-Deltay*D0) + (1+rM)*A0 - pE(jE) + YTransfer(iY);
    else
        r = rM + ((A0/ASmooth)/(1+(A0/ASmooth)^2)+0.5)*(rP-rM);
        TotalIncome = Y0*(1-Deltay*D0) + (1+r)*A0 - pE(jE) + YTransfer(iY);
    end
    w = Wstar - (1-delta)*(1-DeltaW)*W0;
    if w <= -wSmooth
        winvest = kappaZ;
    elseif w > wSmooth
        winvest = 1;
    else
        winvest = kappaZ + ((w/wSmooth)/(1+(w/wSmooth)^2)+0.5)*(1-kappaZ);
    end
    c = TotalIncome - w*winvest - Astar - pFactor * E0 * Wstar * Pstar^(1+phiP);
    
    if c <= lowC
        Astar = Agrid(1);
        jAstar = 0;
        for kA = 2:nA
            if Agrid(kA) > Astar
                jAstar = kA - 1;
                break
            elseif kA == nA
                jAstar = nA-1;
            end
        end
        
        Wstar = lowW + 2e-6;
        jWstar = 0;
        for kW = 2:nW
            if Wgrid(kW) > Wstar
                jWstar = kW - 1;
                break
            elseif kW == nW
                jWstar = nW-1;
            end
        end
        
        Pstar = 0.0;
        jPstar = 0;
        for kW = 2:nW
            if Wgrid(kW) > Pstar*Wstar
                jPstar = kW - 1;
                break
            elseif kW == nW
                jPstar = nW-1;
            end
        end
        tAstar = (Astar - Agrid(jAstar))/(Agrid(jAstar+1)-Agrid(jAstar));
        tWstar = (Wstar - Wgrid(jWstar))/(Wgrid(jWstar+1)-Wgrid(jWstar));
        tPstar = (Pstar*Wstar - Wgrid(jPstar))/(Wgrid(jPstar+1)-Wgrid(jPstar));
        
        w = Wstar - (1-delta)*(1-DeltaW)*W0;
        if w <= -wSmooth
            winvest = kappaZ;
        elseif w > wSmooth
            winvest = 1;
        else
            winvest = kappaZ + ((w/wSmooth)/(1+(w/wSmooth)^2)+0.5)*(1-kappaZ);
        end
        c = TotalIncome - w*winvest - Astar - pFactor * E0 * Wstar * Pstar^(1+phiP);
        
        if c < lowC + 1e-6
            GlobalStop = 1;
        else
            GlobalStop = 0;
            
        end
    else
        GlobalStop = 0;
    end
        
    if c <= lowC || Wstar <= lowW
        ValueStar = -1e9;
    else
        Utility1 = (((c-lowC)/scalefactor)^(1-gamma) - 1)/(1-gamma) + theta * ( ((Wstar-lowW)/scalefactor)^(1-beta) - 1)/(1-beta);
        V0C1 = 0;
        V0C2 = 0;
        for jY = 1:nY
            V0C1 = V0C1 + TransY(iY,jY) * ( (1-tAstar)*((1-tWstar)*v0(jE, jAstar  ,jWstar ,jY,1)   ...
                +  tWstar *v0(jE, jAstar  ,jWstar+1,jY,1))  ...
                + tAstar*((1-tWstar)*v0(jE, jAstar+1,jWstar  ,jY,1)   ...
                +  tWstar *v0(jE, jAstar+1,jWstar+1,jY,1)));
            V0C2 = V0C2 + TransY(iY,jY) * ( (1-tAstar)*((1-tPstar)*v0(jE, jAstar  ,jPstar ,jY,2)   ...
                +  tPstar *v0(jE, jAstar  ,jPstar+1    ,jY,2))  ...
                + tAstar*((1-tPstar)*v0(jE, jAstar+1,jPstar      ,jY,2)   ...
                +  tPstar *v0(jE, jAstar+1,jPstar+1    ,jY,2)));
        end
        ValueStar = Utility1 + rho2 * (a*E0*V0C2 + (1-a*E0)*V0C1);
    end
        
    iterFine = 0;
    stepsize = 1e-3;       
    
    while GlobalStop == 0 && iterFine < 1000
        %%%Calculate Gradients
        
        w = Wstar - (1-delta)*(1-DeltaW)*W0;
        if w <= -wSmooth
            winvest = kappaZ;
        elseif w > wSmooth
            winvest = 1;
        else
            winvest = kappaZ + ((w/wSmooth)/(1+(w/wSmooth)^2)+0.5)*(1-kappaZ);
        end
        c = TotalIncome - w*winvest - Astar - pFactor * E0 * Wstar * Pstar^(1+phiP);
        if c > lowC
            dUdc = ((c-lowC)/scalefactor)^(-gamma)/scalefactor;
        else
            dUdc = 1e9;
        end
        if Wstar > lowW
            dUdW = theta * ((Wstar-lowW)/scalefactor)^(-beta)/scalefactor;
        else
            dUdW = 1e9;
        end
        dVdAC1 = 0;
        dVdAC2 = 0;
        if tAstar <= 0.5 
            if jAstar == 1
                for jY = 1:nY
                    dVdAC1 = dVdAC1 + TransY(iY,jY) * ( ((1-tWstar)*v0(jE, jAstar+1,jWstar  ,jY,1) + tWstar*v0(jE, jAstar+1,jWstar+1,jY,1))  ...
                                                      - ((1-tWstar)*v0(jE, jAstar  ,jWstar  ,jY,1) + tWstar*v0(jE, jAstar  ,jWstar+1,jY,1)) );
                    
                    dVdAC2 = dVdAC2 + TransY(iY,jY) * (((1-tPstar)*v0(jE, jAstar+1,jPstar  ,jY,2) + tPstar*v0(jE, jAstar+1,jPstar+1,jY,2))  ...
                                                     - ((1-tPstar)*v0(jE, jAstar  ,jPstar  ,jY,2) + tPstar*v0(jE, jAstar  ,jPstar+1,jY,2)));
                end
                dVdA = ((1-a*E0)*dVdAC1 + a*E0*dVdAC2)/(Agrid(jAstar+1)-Agrid(jAstar));
            else
                tAstar2 = 0.5 + tAstar;
                for jY = 1:nY
                    dVdAC1 = dVdAC1 + TransY(iY,jY) * ( (1-tAstar2)*((1-tWstar)*v0(jE, jAstar  ,jWstar ,jY,1)   ...
                                                                     +  tWstar *v0(jE, jAstar  ,jWstar+1,jY,1))  ...
                                                          + tAstar2*((1-tWstar)*v0(jE, jAstar+1,jWstar  ,jY,1)   ...
                                                                     +  tWstar *v0(jE, jAstar+1,jWstar+1,jY,1)) - ...
                                                       ((1-tAstar2)*((1-tWstar)*v0(jE, jAstar-1,jWstar ,jY,1)   ...
                                                                     +  tWstar *v0(jE, jAstar-1,jWstar+1,jY,1))  ...
                                                          + tAstar2*((1-tWstar)*v0(jE, jAstar  ,jWstar  ,jY,1)   ...
                                                                     +  tWstar *v0(jE, jAstar  ,jWstar+1,jY,1))));
                    dVdAC2 = dVdAC2 + TransY(iY,jY) * ( (1-tAstar2)*((1-tPstar)*v0(jE, jAstar  ,jPstar  ,jY,2)   ...
                                                                     +  tPstar *v0(jE, jAstar  ,jPstar+1,jY,2))  ...
                                                          + tAstar2*((1-tPstar)*v0(jE, jAstar+1,jPstar  ,jY,2)   ...
                                                                     +  tPstar *v0(jE, jAstar+1,jPstar+1,jY,2)) - ...
                                                       ((1-tAstar2)*((1-tPstar)*v0(jE, jAstar-1,jPstar  ,jY,2)   ...
                                                                     +  tPstar *v0(jE, jAstar-1,jPstar+1,jY,2))  ...
                                                          + tAstar2*((1-tPstar)*v0(jE, jAstar  ,jPstar  ,jY,2)   ...
                                                                     +  tPstar *v0(jE, jAstar  ,jPstar+1,jY,2))));
                end
                dVdA = ((1-a*E0)*dVdAC1 + a*E0*dVdAC2)/(  ((1-tAstar2)*Agrid(jAstar)+tAstar2*Agrid(jAstar+1)) - ((1-tAstar2)*Agrid(jAstar-1)+tAstar2*Agrid(jAstar)) );
            end
        else
            if jAstar == nA-1
%                 for jY = 1:nY
%                     dVdAC1 = dVdAC1 + TransY(iY,jY) * (((1-tWstar)*v0(jE, jAstar+1,jWstar  ,jY,1) + tWstar*v0(jE, jAstar+1,jWstar+1,jY,1))  ...
%                                                      - ((1-tWstar)*v0(jE, jAstar  ,jWstar  ,jY,1) + tWstar*v0(jE, jAstar  ,jWstar+1,jY,1)));
%                     
%                     dVdAC2 = dVdAC2 + TransY(iY,jY) * (((1-tPstar)*v0(jE, jAstar+1,jPstar  ,jY,2) + tPstar*v0(jE, jAstar+1,jPstar+1,jY,2))  ...
%                                                      - ((1-tPstar)*v0(jE, jAstar  ,jPstar  ,jY,2) + tPstar*v0(jE, jAstar  ,jPstar+1,jY,2)));
%                 end
                tAstar2 = -0.5 + tAstar;
                for jY = 1:nY
                    dVdAC1 = dVdAC1 + TransY(iY,jY) * ( (1-tAstar2)*((1-tWstar)*v0(jE, jAstar+1,jWstar ,jY,1)   ...
                                                                     +  tWstar *v0(jE, jAstar+1,jWstar+1,jY,1))  ...
                                                          + tAstar2*((1-tWstar)*v0(jE, jAstar+1,jWstar  ,jY,1)   ...
                                                                     +  tWstar *v0(jE, jAstar+1,jWstar+1,jY,1)) - ...
                                                       ((1-tAstar2)*((1-tWstar)*v0(jE, jAstar  ,jWstar ,jY,1)   ...
                                                                     +  tWstar *v0(jE, jAstar  ,jWstar+1,jY,1))  ...
                                                          + tAstar2*((1-tWstar)*v0(jE, jAstar+1,jWstar  ,jY,1)   ...
                                                                     +  tWstar *v0(jE, jAstar+1,jWstar+1,jY,1))));
                    dVdAC2 = dVdAC2 + TransY(iY,jY) * ( (1-tAstar2)*((1-tPstar)*v0(jE, jAstar+1,jPstar  ,jY,2)   ...
                                                                     +  tPstar *v0(jE, jAstar+1,jPstar+1,jY,2))  ...
                                                          + tAstar2*((1-tPstar)*v0(jE, jAstar+1,jPstar  ,jY,2)   ...
                                                                     +  tPstar *v0(jE, jAstar+1,jPstar+1,jY,2)) - ...
                                                       ((1-tAstar2)*((1-tPstar)*v0(jE, jAstar  ,jPstar  ,jY,2)   ...
                                                                     +  tPstar *v0(jE, jAstar  ,jPstar+1,jY,2))  ...
                                                          + tAstar2*((1-tPstar)*v0(jE, jAstar+1,jPstar  ,jY,2)   ...
                                                                     +  tPstar *v0(jE, jAstar+1,jPstar+1,jY,2))));
                end
                dVdA = ((1-a*E0)*dVdAC1 + a*E0*dVdAC2)/(Agrid(jAstar+1)-Agrid(jAstar));
            else
                tAstar2 = -0.5 + tAstar;
                for jY = 1:nY
                    dVdAC1 = dVdAC1 + TransY(iY,jY) * ( (1-tAstar2)*((1-tWstar)*v0(jE, jAstar+1,jWstar ,jY,1)   ...
                                                                     +  tWstar *v0(jE, jAstar+1,jWstar+1,jY,1))  ...
                                                          + tAstar2*((1-tWstar)*v0(jE, jAstar+2,jWstar  ,jY,1)   ...
                                                                     +  tWstar *v0(jE, jAstar+2,jWstar+1,jY,1)) - ...
                                                       ((1-tAstar2)*((1-tWstar)*v0(jE, jAstar  ,jWstar ,jY,1)   ...
                                                                     +  tWstar *v0(jE, jAstar  ,jWstar+1,jY,1))  ...
                                                          + tAstar2*((1-tWstar)*v0(jE, jAstar+1,jWstar  ,jY,1)   ...
                                                                     +  tWstar *v0(jE, jAstar+1,jWstar+1,jY,1))));
                    dVdAC2 = dVdAC2 + TransY(iY,jY) * ( (1-tAstar2)*((1-tPstar)*v0(jE, jAstar+1,jPstar  ,jY,2)   ...
                                                                     +  tPstar *v0(jE, jAstar+1,jPstar+1,jY,2))  ...
                                                          + tAstar2*((1-tPstar)*v0(jE, jAstar+2,jPstar  ,jY,2)   ...
                                                                     +  tPstar *v0(jE, jAstar+2,jPstar+1,jY,2)) - ...
                                                       ((1-tAstar2)*((1-tPstar)*v0(jE, jAstar  ,jPstar  ,jY,2)   ...
                                                                     +  tPstar *v0(jE, jAstar  ,jPstar+1,jY,2))  ...
                                                          + tAstar2*((1-tPstar)*v0(jE, jAstar+1,jPstar  ,jY,2)   ...
                                                                     +  tPstar *v0(jE, jAstar+1,jPstar+1,jY,2))));
                end
                dVdA = ((1-a*E0)*dVdAC1 + a*E0*dVdAC2)/(  ((1-tAstar2)*Agrid(jAstar+1)+tAstar2*Agrid(jAstar+2)) - ((1-tAstar2)*Agrid(jAstar)+tAstar2*Agrid(jAstar+1)) );
            end 
        end
        GradA = rho2*dVdA - dUdc;
        
        dVdWC1 = 0;
        dVdWC2 = 0;
        if tWstar <= 0.5
            if jWstar == 1
                for jY = 1:nY
                    dVdWC1 = dVdWC1 + TransY(iY,jY) * (((1-tAstar)*v0(jE, jAstar,jWstar+1,jY,1) + tAstar*v0(jE, jAstar+1,jWstar+1,jY,1))  ...
                                                     - ((1-tAstar)*v0(jE, jAstar,jWstar  ,jY,1) + tAstar*v0(jE, jAstar+1,jWstar  ,jY,1)));
                end
                dVdWC1 = (1-a*E0)*dVdWC1 /(Wgrid(jWstar+1)-Wgrid(jWstar));
            else
                tWstar2 = 0.5 + tWstar;
                for jY = 1:nY
                    dVdWC1 = dVdWC1 + TransY(iY,jY) * ( (1-tAstar)*((1-tWstar2)*v0(jE, jAstar  ,jWstar  ,jY,1)   ...
                                                                    +  tWstar2 *v0(jE, jAstar  ,jWstar+1,jY,1))  ...
                                                          + tAstar*((1-tWstar2)*v0(jE, jAstar+1,jWstar  ,jY,1)   ...
                                                                     + tWstar2 *v0(jE, jAstar+1,jWstar+1,jY,1)) - ...
                                                       ((1-tAstar)*((1-tWstar2)*v0(jE, jAstar  ,jWstar-1,jY,1)   ...
                                                                    +  tWstar2 *v0(jE, jAstar  ,jWstar  ,jY,1))  ...
                                                          + tAstar*((1-tWstar2)*v0(jE, jAstar+1,jWstar-1,jY,1)   ...
                                                                    +  tWstar2 *v0(jE, jAstar+1,jWstar  ,jY,1))));
                end
                dVdWC1 = (1-a*E0)*dVdWC1 /(  ((1-tWstar2)*Wgrid(jWstar)+tWstar2*Wgrid(jWstar+1)) - ((1-tWstar2)*Wgrid(jWstar-1)+tWstar2*Wgrid(jWstar)) );
            end
        else
            if jWstar == nW - 1
%                 for jY = 1:nY
%                     dVdWC1 = dVdWC1 + TransY(iY,jY) * (((1-tAstar)*v0(jE, jAstar,jWstar+1,jY,1) + tAstar*v0(jE, jAstar+1,jWstar+1,jY,1))  ...
%                                                      - ((1-tAstar)*v0(jE, jAstar,jWstar  ,jY,1) + tAstar*v0(jE, jAstar+1,jWstar  ,jY,1)));
%                 end
                tWstar2 = -0.5 + tWstar;
                for jY = 1:nY
                    dVdWC1 = dVdWC1 + TransY(iY,jY) * ( (1-tAstar)*((1-tWstar2)*v0(jE, jAstar  ,jWstar+1,jY,1)   ...
                                                                    +  tWstar2 *v0(jE, jAstar  ,jWstar+1,jY,1))  ...
                                                        +   tAstar*((1-tWstar2)*v0(jE, jAstar+1,jWstar+1,jY,1)   ...
                                                                    +  tWstar2 *v0(jE, jAstar+1,jWstar+1,jY,1)) - ...
                                                       ((1-tAstar)*((1-tWstar2)*v0(jE, jAstar  ,jWstar  ,jY,1)   ...
                                                                    +  tWstar2 *v0(jE, jAstar  ,jWstar+1,jY,1))  ...
                                                          + tAstar*((1-tWstar2)*v0(jE, jAstar+1,jWstar  ,jY,1)   ...
                                                                    +  tWstar2 *v0(jE, jAstar+1,jWstar+1,jY,1))));
                end
                dVdWC1 = (1-a*E0)*dVdWC1 /(Wgrid(jWstar+1)-Wgrid(jWstar));
            else
                tWstar2 = -0.5 + tWstar;
                for jY = 1:nY
                    dVdWC1 = dVdWC1 + TransY(iY,jY) * ( (1-tAstar)*((1-tWstar2)*v0(jE, jAstar  ,jWstar+1,jY,1)   ...
                                                                    +  tWstar2 *v0(jE, jAstar  ,jWstar+2,jY,1))  ...
                                                        +   tAstar*((1-tWstar2)*v0(jE, jAstar+1,jWstar+1,jY,1)   ...
                                                                    +  tWstar2 *v0(jE, jAstar+1,jWstar+2,jY,1)) - ...
                                                       ((1-tAstar)*((1-tWstar2)*v0(jE, jAstar  ,jWstar  ,jY,1)   ...
                                                                    +  tWstar2 *v0(jE, jAstar  ,jWstar+1,jY,1))  ...
                                                          + tAstar*((1-tWstar2)*v0(jE, jAstar+1,jWstar  ,jY,1)   ...
                                                                    +  tWstar2 *v0(jE, jAstar+1,jWstar+1,jY,1))));
                end
                dVdWC1 = (1-a*E0)*dVdWC1 /(  ((1-tWstar2)*Wgrid(jWstar+1)+tWstar2*Wgrid(jWstar+2)) - ((1-tWstar2)*Wgrid(jWstar)+tWstar2*Wgrid(jWstar+1)) );
            end
        end
           
        if tPstar <= 0.5
            if jPstar == 1
                for jY = 1:nY
                    dVdWC2 = dVdWC2 + TransY(iY,jY) * (((1-tAstar)*v0(jE, jAstar,jPstar+1,jY,2) + tAstar*v0(jE, jAstar+1,jPstar+1,jY,2))  ...
                                                     - ((1-tAstar)*v0(jE, jAstar,jPstar  ,jY,2) + tAstar*v0(jE, jAstar+1,jPstar  ,jY,2)));
                end
                dVdWC2 = a*E0*dVdWC2 /(Wgrid(jPstar+1)-Wgrid(jPstar));
            else
                tPstar2 = 0.5 + tPstar;
                for jY = 1:nY
                    dVdWC2 = dVdWC2 + TransY(iY,jY) * ( (1-tAstar)*((1-tPstar2)*v0(jE, jAstar  ,jPstar  ,jY,2)   ...
                                                                    +  tPstar2 *v0(jE, jAstar  ,jPstar+1,jY,2))  ...
                                                       +    tAstar*((1-tPstar2)*v0(jE, jAstar+1,jPstar  ,jY,2)   ...
                                                                    +  tPstar2 *v0(jE, jAstar+1,jPstar+1,jY,2)) - ...
                                                       ((1-tAstar)*((1-tPstar2)*v0(jE, jAstar  ,jPstar-1,jY,2)   ...
                                                                    +  tPstar2 *v0(jE, jAstar  ,jPstar  ,jY,2))  ...
                                                          + tAstar*((1-tPstar2)*v0(jE, jAstar+1,jPstar-1,jY,2)   ...
                                                                    +  tPstar2 *v0(jE, jAstar+1,jPstar  ,jY,2))));
                end
                dVdWC2 = a*E0*dVdWC2 /(  ((1-tPstar2)*Wgrid(jPstar)+tPstar2*Wgrid(jPstar+1)) - ((1-tPstar2)*Wgrid(jPstar-1)+tPstar2*Wgrid(jPstar)) );
            end
        else
            if jPstar == nW - 1
%                 for jY = 1:nY
%                     dVdWC2 = dVdWC2 + TransY(iY,jY) * (((1-tAstar)*v0(jE, jAstar,jPstar+1,jY,2) + tAstar*v0(jE, jAstar+1,jPstar+1,jY,2))  ...
%                                                      - ((1-tAstar)*v0(jE, jAstar,jPstar  ,jY,2) + tAstar*v0(jE, jAstar+1,jPstar  ,jY,2)));
%                 end
                tPstar2 = -0.5 + tPstar;
                for jY = 1:nY
                    dVdWC2 = dVdWC2 + TransY(iY,jY) * ( (1-tAstar)*((1-tPstar2)*v0(jE, jAstar  ,jPstar+1,jY,2)   ...
                                                                    +  tPstar2 *v0(jE, jAstar  ,jPstar+1,jY,2))  ...
                                                       +    tAstar*((1-tPstar2)*v0(jE, jAstar+1,jPstar+1,jY,2)   ...
                                                                    +  tPstar2 *v0(jE, jAstar+1,jPstar+1,jY,2)) - ...
                                                       ((1-tAstar)*((1-tPstar2)*v0(jE, jAstar  ,jPstar  ,jY,2)   ...
                                                                    +  tPstar2 *v0(jE, jAstar  ,jPstar+1,jY,2))  ...
                                                          + tAstar*((1-tPstar2)*v0(jE, jAstar+1,jPstar  ,jY,2)   ...
                                                                    +  tPstar2 *v0(jE, jAstar+1,jPstar+1,jY,2))));
                end
                dVdWC2 = a*E0*dVdWC2 /(Wgrid(jPstar+1)-Wgrid(jPstar));
            else
                tPstar2 = -0.5 + tPstar;
                for jY = 1:nY
                    dVdWC2 = dVdWC2 + TransY(iY,jY) * ( (1-tAstar)*((1-tPstar2)*v0(jE, jAstar  ,jPstar+1,jY,2)   ...
                                                                    +  tPstar2 *v0(jE, jAstar  ,jPstar+2,jY,2))  ...
                                                       +    tAstar*((1-tPstar2)*v0(jE, jAstar+1,jPstar+1,jY,2)   ...
                                                                    +  tPstar2 *v0(jE, jAstar+1,jPstar+2,jY,2)) - ...
                                                       ((1-tAstar)*((1-tPstar2)*v0(jE, jAstar  ,jPstar  ,jY,2)   ...
                                                                    +  tPstar2 *v0(jE, jAstar  ,jPstar+1,jY,2))  ...
                                                          + tAstar*((1-tPstar2)*v0(jE, jAstar+1,jPstar  ,jY,2)   ...
                                                                    +  tPstar2 *v0(jE, jAstar+1,jPstar+1,jY,2))));
                end
                dVdWC2 = a*E0*dVdWC2 /(  ((1-tPstar2)*Wgrid(jPstar+1)+tPstar2*Wgrid(jPstar+2)) - ((1-tPstar2)*Wgrid(jPstar)+tPstar2*Wgrid(jPstar+1)) );
            end
        end
        
        dVdW = dVdWC1 + Pstar*a*E0*dVdWC2;
        GradW = rho2*dVdW - dUdc*(winvest + pFactor*E0*Pstar^(1+phiP)) + dUdW;
        
        dVdP = Wstar*a*E0*dVdWC2;
        GradP = rho2*dVdP - dUdc*pFactor*E0*Wstar*(1+phiP)*max(1e-9,Pstar)^phiP;
        
        %% Scale gradient to 1
        
        if Astar <= Agrid(1) && GradA < 0
            GradA = 0;
        end
        if Wstar <= lowW + 1e-6 && GradW < 0
            GradW = 0;
        end
        if (Pstar <= 0.0 && GradP < 0) || (Pstar >= 1.0 && GradP > 0.0)
            GradP = 0;
        end
        
        GradNorm = sqrt(GradA^2+GradW^2+GradP^2);
        GradAnew = GradA/GradNorm;
        GradWnew = GradW/GradNorm;
        GradPnew = GradP/GradNorm;
        
        if GradNorm > 1e-6
            %% Make step in Gradient direction
            LineStepStop = 0;
        
            while LineStepStop == 0
                
                Anew = Astar + stepsize*GradAnew;
                Anew = max(Agrid(1),min(Agrid(nA),Anew));
                Wnew = Wstar + stepsize*GradWnew;
                Wnew = max(lowW+1e-6,min(Wgrid(nW),Wnew));
                Pnew = Pstar + stepsize*GradPnew;
                Pnew = max(0.0,min(1.0,Pnew));
                
                jAnew = 0;
                for kA = 2:nA
                    if Agrid(kA) > Anew
                        jAnew = kA - 1;
                        break
                    elseif kA == nA
                        jAnew = nA-1;
                    end
                end
                
                jWnew = 0;
                for kW = 2:nW
                    if Wgrid(kW) > Wnew
                        jWnew = kW - 1;
                        break
                    elseif kW == nW
                        jWnew = nW-1;
                    end
                end
                
                jPnew = 0;
                for kW = 2:nW
                    if Wgrid(kW) > Pnew*Wnew
                        jPnew = kW - 1;
                        break
                    elseif kW == nW
                        jPnew = nW-1;
                    end
                end
                tAnew = (Anew - Agrid(jAnew))/(Agrid(jAnew+1)-Agrid(jAnew));
                tWnew = (Wnew - Wgrid(jWnew))/(Wgrid(jWnew+1)-Wgrid(jWnew));
                tPnew = (Pnew*Wnew - Wgrid(jPnew))/(Wgrid(jPnew+1)-Wgrid(jPnew));
                
                w = Wnew - (1-delta)*(1-DeltaW)*W0;
                if w <= -wSmooth
                    winvest = kappaZ;
                elseif w > wSmooth
                    winvest = 1;
                else
                    winvest = kappaZ + ((w/wSmooth)/(1+(w/wSmooth)^2)+0.5)*(1-kappaZ);
                end
                c = TotalIncome - w*winvest - Anew - pFactor * E0 * Wnew * Pnew^(1+phiP);
                
                if c <= lowC || Wnew <= lowW
                    ValueStarNew = -1e9;
                else
                    Utility1 = (((c-lowC)/scalefactor)^(1-gamma) - 1)/(1-gamma) + theta * ( ((Wnew-lowW)/scalefactor)^(1-beta) - 1)/(1-beta);
                    V0C1 = 0;
                    V0C2 = 0;
                    for jY = 1:nY
                        V0C1 = V0C1 + TransY(iY,jY) * ( (1-tAnew)*((1-tWnew)*v0(jE, jAnew  ,jWnew ,jY,1)   ...
                            +  tWnew *v0(jE, jAnew  ,jWnew+1,jY,1))  ...
                            + tAnew*((1-tWnew)*v0(jE, jAnew+1,jWnew  ,jY,1)   ...
                            +  tWnew *v0(jE, jAnew+1,jWnew+1,jY,1)));
                        V0C2 = V0C2 + TransY(iY,jY) * ( (1-tAnew)*((1-tPnew)*v0(jE, jAnew  ,jPnew ,jY,2)   ...
                            +  tPnew *v0(jE, jAnew  ,jPnew+1    ,jY,2))  ...
                            + tAnew*((1-tPnew)*v0(jE, jAnew+1,jPnew      ,jY,2)   ...
                            +  tPnew *v0(jE, jAnew+1,jPnew+1    ,jY,2)));
                    end
                    ValueStarNew = Utility1 + rho2 * (a*E0*V0C2 + (1-a*E0)*V0C1);
                end
                
                if ValueStarNew > ValueStar
                    LineStepStop = 1;
                    stepsize = min(1e-2,stepsize*1.25);
                    
                    Astar = Anew;
                    Wstar = Wnew;
                    Pstar = Pnew;
                    ValueStar = ValueStarNew;
                    jAstar = jAnew;
                    tAstar = tAnew;
                    jWstar = jWnew;
                    tWstar = tWnew;
                    jPstar = jPnew;
                    tPstar = tPnew;
%                    fprintf(" + | stepsize = %g | GradNorm = %g | ValueStar = %g \n",stepsize,GradNorm,ValueStar)
                elseif stepsize < 1e-9
                    LineStepStop = 1;
                    GlobalStop = 1;
                else
                    stepsize = stepsize*0.5;
 %                   fprintf(" - | stepsize = %g | GradNorm = %g | ValueStar = %g \n",stepsize,GradNorm,ValueStar)
                end
            end
            iterFine = iterFine + 1;
          
        else
            GlobalStop = 1;
        end        
    end
    PolA = Astar;
    PolW = Wstar;
    PolP = Pstar;
    Value = ValueStar;
end


function [Value,PolA,PolW,PolP,iterFine,GradNorm] = OptiSearchStayFineGrad(iE,iA,iW,iY,iD,StartA,StartW,StartP)

    E0 = Egrid(iE);
    jE = iE;
    A0 = Agrid(iA);
    W0 = Wgrid(iW);
    Y0 = Ygrid(iY);
    D0 = Dgrid(iD);
    
    GradNorm = 0;
    Anew = 0; Wnew = 0; Pnew = 0;
        
    Astar = StartA;
    jAstar = 0;
    for kA = 2:nA
        if Agrid(kA) > Astar
            jAstar = kA - 1;
            break
        elseif kA == nA
            jAstar = nA-1;
        end
    end
    
    Wstar = StartW;
    jWstar = 0;
    for kW = 2:nW
        if Wgrid(kW) > Wstar
            jWstar = kW - 1;
            break
        elseif kW == nW
            jWstar = nW-1;
        end
    end
    
    Pstar = StartP;
    jPstar = 0;
    for kW = 2:nW
        if Wgrid(kW) > Pstar*Wstar
            jPstar = kW - 1;
            break
        elseif kW == nW
            jPstar = nW-1;
        end
    end
    tAstar = (Astar - Agrid(jAstar))/(Agrid(jAstar+1)-Agrid(jAstar));
    tWstar = (Wstar - Wgrid(jWstar))/(Wgrid(jWstar+1)-Wgrid(jWstar));
    tPstar = (Pstar*Wstar - Wgrid(jPstar))/(Wgrid(jPstar+1)-Wgrid(jPstar));
        
    if A0 > ASmooth
        TotalIncome = Y0*(1-Deltay*D0) + (1+rP)*A0 - pE(jE) + YTransfer(iY);
    elseif A0 <= -ASmooth
        TotalIncome = Y0*(1-Deltay*D0) + (1+rM)*A0 - pE(jE) + YTransfer(iY);
    else
        r = rM + ((A0/ASmooth)/(1+(A0/ASmooth)^2)+0.5)*(rP-rM);
        TotalIncome = Y0*(1-Deltay*D0) + (1+r)*A0 - pE(jE) + YTransfer(iY);
    end
    w = Wstar - (1-delta)*W0;
    if w <= -wSmooth
        winvest = kappaZ;
    elseif w > wSmooth
        winvest = 1;
    else
        winvest = kappaZ + ((w/wSmooth)/(1+(w/wSmooth)^2)+0.5)*(1-kappaZ);
    end
    c = TotalIncome - w*winvest - Astar - pFactor * E0 * Wstar * Pstar^(1+phiP);
    if c <= lowC
        Astar = Agrid(1);
        jAstar = 0;
        for kA = 2:nA
            if Agrid(kA) > Astar
                jAstar = kA - 1;
                break
            elseif kA == nA
                jAstar = nA-1;
            end
        end
        
        Wstar = lowW + 2e-6;
        jWstar = 0;
        for kW = 2:nW
            if Wgrid(kW) > Wstar
                jWstar = kW - 1;
                break
            elseif kW == nW
                jWstar = nW-1;
            end
        end
        
        Pstar = 0.0;
        jPstar = 0;
        for kW = 2:nW
            if Wgrid(kW) > Pstar*Wstar
                jPstar = kW - 1;
                break
            elseif kW == nW
                jPstar = nW-1;
            end
        end
        tAstar = (Astar - Agrid(jAstar))/(Agrid(jAstar+1)-Agrid(jAstar));
        tWstar = (Wstar - Wgrid(jWstar))/(Wgrid(jWstar+1)-Wgrid(jWstar));
        tPstar = (Pstar*Wstar - Wgrid(jPstar))/(Wgrid(jPstar+1)-Wgrid(jPstar));
        
        w = Wstar - (1-delta)*W0;
        if w <= -wSmooth
            winvest = kappaZ;
        elseif w > wSmooth
            winvest = 1;
        else
            winvest = kappaZ + ((w/wSmooth)/(1+(w/wSmooth)^2)+0.5)*(1-kappaZ);
        end
        c = TotalIncome - w*winvest - Astar - pFactor * E0 * Wstar * Pstar^(1+phiP);
        
        if c < lowC + 1e-6
            GlobalStop = 1;
        else
            GlobalStop = 0;
        end 
    else
        GlobalStop = 0;
    end
    
    if c <= lowC || Wstar <= lowW
        ValueStar = -1e9;
    else
        Utility1 = (((c-lowC)/scalefactor)^(1-gamma) - 1)/(1-gamma) + theta * ( ((Wstar-lowW)/scalefactor)^(1-beta) - 1)/(1-beta);
        V0C1 = 0;
        V0C2 = 0;
        for jY = 1:nY
            V0C1 = V0C1 + TransY(iY,jY) * ( (1-tAstar)*((1-tWstar)*v0(jE, jAstar  ,jWstar ,jY,1)   ...
                +  tWstar *v0(jE, jAstar  ,jWstar+1,jY,1))  ...
                + tAstar*((1-tWstar)*v0(jE, jAstar+1,jWstar  ,jY,1)   ...
                +  tWstar *v0(jE, jAstar+1,jWstar+1,jY,1)));
            V0C2 = V0C2 + TransY(iY,jY) * ( (1-tAstar)*((1-tPstar)*v0(jE, jAstar  ,jPstar ,jY,2)   ...
                +  tPstar *v0(jE, jAstar  ,jPstar+1    ,jY,2))  ...
                + tAstar*((1-tPstar)*v0(jE, jAstar+1,jPstar      ,jY,2)   ...
                +  tPstar *v0(jE, jAstar+1,jPstar+1    ,jY,2)));
        end
        ValueStar = Utility1 + rho2 * (a*E0*V0C2 + (1-a*E0)*V0C1);
    end
    
    iterFine = 0;
    stepsize = 1e-1;
    
    while GlobalStop == 0 && iterFine < 500
        %%%Calculate Gradients
        
        w = Wstar - (1-delta)*W0;
        if w <= -wSmooth
            winvest = kappaZ;
        elseif w > wSmooth
            winvest = 1;
        else
            winvest = kappaZ + ((w/wSmooth)/(1+(w/wSmooth)^2)+0.5)*(1-kappaZ);
        end
        c = TotalIncome - w*winvest - Astar - pFactor * E0 * Wstar * Pstar^(1+phiP);
        if c > lowC
            dUdc = ((c-lowC)/scalefactor)^(-gamma)/scalefactor;
        else
            dUdc = 1e9;
        end
        if c > lowC
            dUdW = theta * ((Wstar-lowW)/scalefactor)^(-beta)/scalefactor;
        else
            dUdW = 1e9;
        end
        dVdAC1 = 0;
        dVdAC2 = 0;
        for jY = 1:nY
            dVdAC1 = dVdAC1 + TransY(iY,jY) * (((1-tWstar)*v0(jE, jAstar+1,jWstar  ,jY,1) + tWstar*v0(jE, jAstar+1,jWstar+1,jY,1))  ...
                - ((1-tWstar)*v0(jE, jAstar  ,jWstar  ,jY,1) + tWstar *v0(jE, jAstar  ,jWstar+1,jY,1)));
            
            dVdAC2 = dVdAC2 + TransY(iY,jY) * (((1-tPstar)*v0(jE, jAstar+1,jPstar  ,jY,2) + tPstar*v0(jE, jAstar+1,jPstar+1,jY,2))  ...
                - ((1-tPstar)*v0(jE, jAstar  ,jPstar  ,jY,2) + tPstar *v0(jE, jAstar  ,jPstar+1,jY,2)));
        end
        dVdA = ((1-a*E0)*dVdAC1 + a*E0*dVdAC2)/(Agrid(jAstar+1)-Agrid(jAstar));
        GradA = rho2*dVdA - dUdc;
        
        dVdWC1 = 0;
        dVdWC2 = 0;
        for jY = 1:nY
            dVdWC1 = dVdWC1 + TransY(iY,jY) * (((1-tAstar)*v0(jE, jAstar,jWstar+1,jY,1) + tAstar *v0(jE, jAstar+1,jWstar+1,jY,1))  ...
                - ((1-tAstar)*v0(jE, jAstar  ,jWstar ,jY,1) + tAstar *v0(jE, jAstar+1 ,jWstar,jY,1)));
            dVdWC2 = dVdWC2 + TransY(iY,jY) * (((1-tAstar)*v0(jE, jAstar,jPstar+1,jY,2) + tAstar *v0(jE, jAstar+1,jPstar+1,jY,2))  ...
                - ((1-tAstar)*v0(jE, jAstar  ,jPstar ,jY,2) + tAstar *v0(jE, jAstar+1 ,jPstar,jY,2)));
        end
        dVdW = (1-a*E0)*dVdWC1/(Wgrid(jWstar+1)-Wgrid(jWstar)) + Pstar*a*E0*dVdWC2/(Wgrid(jPstar+1)-Wgrid(jPstar));
        GradW = rho2*dVdW - dUdc*(winvest + pFactor*E0*Pstar^(1+phiP)) + dUdW;
        
        dVdP = Wstar*a*E0*dVdWC2/(Wgrid(jPstar+1)-Wgrid(jPstar));
        GradP = rho2*dVdP - dUdc*pFactor*E0*Wstar*(1+phiP)*Pstar^phiP;
        
        %% Scale gradient to 1
        if Astar <= Agrid(1) && GradA < 0
            GradA = 0;
        end
        if Wstar <= lowW + 1e-6 && GradW < 0
            GradW = 0;
        end
        if (Pstar <= 0.0 && GradP < 0) || (Pstar >= 1.0 && GradP > 0)
            GradP = 0;
        end
        GradNorm = sqrt(GradA^2+GradW^2+GradP^2);
        GradAnew = GradA/GradNorm;
        GradWnew = GradW/GradNorm;
        GradPnew = GradP/GradNorm;
        
        
        if GradNorm > 1e-6
            %% Make step in Gradient direction
            LineStepStop = 0;
        
            while LineStepStop == 0
                
                Anew = Astar + stepsize*GradAnew;
                Anew = max(Agrid(1),min(Agrid(nA),Anew));
                Wnew = Wstar + stepsize*GradWnew;
                Wnew = max(lowW+1e-6,min(Wgrid(nW),Wnew));
                Pnew = Pstar + stepsize*GradPnew;
                Pnew = max(0.0,min(1.0,Pnew));
                
                jAnew = 0;
                for kA = 2:nA
                    if Agrid(kA) > Anew
                        jAnew = kA - 1;
                        break
                    elseif kA == nA
                        jAnew = nA-1;
                    end
                end
                
                jWnew = 0;
                for kW = 2:nW
                    if Wgrid(kW) > Wnew
                        jWnew = kW - 1;
                        break
                    elseif kW == nW
                        jWnew = nW-1;
                    end
                end
                
                jPnew = 0;
                for kW = 2:nW
                    if Wgrid(kW) > Pnew*Wnew
                        jPnew = kW - 1;
                        break
                    elseif kW == nW
                        jPnew = nW-1;
                    end
                end
                tAnew = (Anew - Agrid(jAnew))/(Agrid(jAnew+1)-Agrid(jAnew));
                tWnew = (Wnew - Wgrid(jWnew))/(Wgrid(jWnew+1)-Wgrid(jWnew));
                tPnew = (Pnew*Wnew - Wgrid(jPnew))/(Wgrid(jPnew+1)-Wgrid(jPnew));
                
                w = Wnew - (1-delta)*W0;
                if w <= -wSmooth
                    winvest = kappaZ;
                elseif w > wSmooth
                    winvest = 1;
                else
                    winvest = kappaZ + ((w/wSmooth)/(1+(w/wSmooth)^2)+0.5)*(1-kappaZ);
                end
                c = TotalIncome - w*winvest - Anew - pFactor * E0 * Wnew * Pnew^(1+phiP);
                
                if c <= lowC || Wnew <= lowW
                    ValueStarNew = -1e9;
                else
                    Utility1 = (((c-lowC)/scalefactor)^(1-gamma) - 1)/(1-gamma) + theta * ( ((Wnew-lowW)/scalefactor)^(1-beta) - 1)/(1-beta);
                    V0C1 = 0;
                    V0C2 = 0;
                    for jY = 1:nY
                        V0C1 = V0C1 + TransY(iY,jY) * ( (1-tAnew)*((1-tWnew)*v0(jE, jAnew  ,jWnew ,jY,1)   ...
                            +  tWnew *v0(jE, jAnew  ,jWnew+1,jY,1))  ...
                            + tAnew*((1-tWnew)*v0(jE, jAnew+1,jWnew  ,jY,1)   ...
                            +  tWnew *v0(jE, jAnew+1,jWnew+1,jY,1)));
                        V0C2 = V0C2 + TransY(iY,jY) * ( (1-tAnew)*((1-tPnew)*v0(jE, jAnew  ,jPnew ,jY,2)   ...
                            +  tPnew *v0(jE, jAnew  ,jPnew+1    ,jY,2))  ...
                            + tAnew*((1-tPnew)*v0(jE, jAnew+1,jPnew      ,jY,2)   ...
                            +  tPnew *v0(jE, jAnew+1,jPnew+1    ,jY,2)));
                    end
                    ValueStarNew = Utility1 + rho2 * (a*E0*V0C2 + (1-a*E0)*V0C1);
                end
                
                if ValueStarNew > ValueStar
                    LineStepStop = 1;
                    stepsize = min(1e1,stepsize*1.25);
                    
                    Astar = Anew;
                    Wstar = Wnew;
                    Pstar = Pnew;
                    ValueStar = ValueStarNew;
                    jAstar = jAnew;
                    tAstar = tAnew;
                    jWstar = jWnew;
                    tWstar = tWnew;
                    jPstar = jPnew;
                    tPstar = tPnew;
%                    fprintf(" + | stepsize = %g | GradNorm = %g | ValueStar = %g \n",stepsize,GradNorm,ValueStar)                    
                elseif stepsize < 1e-9
                    LineStepStop = 1;
                    GlobalStop = 1;
                else
                    stepsize = stepsize*0.5;
%                   fprintf(" - | stepsize = %g | GradNorm = %g | ValueStar = %g \n",stepsize,GradNorm,ValueStar)
                end
            end
            iterFine = iterFine + 1;
          
        else
            GlobalStop = 1;
        end        
    end
    PolA = Astar;
    PolW = Wstar;
    PolP = Pstar;
    Value = ValueStar;
end

function [Value,PolA,PolW,PolP,iterFine,GradNorm] = OptiSearchStayFineGradSmoothed(iE,iA,iW,iY,iD,StartA,StartW,StartP)

    E0 = Egrid(iE);
    jE = iE;
    A0 = Agrid(iA);
    W0 = Wgrid(iW);
    Y0 = Ygrid(iY);
    D0 = Dgrid(iD);
    
    GradNorm = 0;
        
    Astar = StartA;
    jAstar = 0;
    for kA = 2:nA
        if Agrid(kA) > Astar
            jAstar = kA - 1;
            break
        elseif kA == nA
            jAstar = nA-1;
        end
    end
    
    Wstar = StartW;
    jWstar = 0;
    for kW = 2:nW
        if Wgrid(kW) > Wstar
            jWstar = kW - 1;
            break
        elseif kW == nW
            jWstar = nW-1;
        end
    end
    
    Pstar = StartP;
    jPstar = 0;
    for kW = 2:nW
        if Wgrid(kW) > Pstar*Wstar
            jPstar = kW - 1;
            break
        elseif kW == nW
            jPstar = nW-1;
        end
    end
    
    tAstar = (Astar - Agrid(jAstar))/(Agrid(jAstar+1)-Agrid(jAstar));
    tWstar = (Wstar - Wgrid(jWstar))/(Wgrid(jWstar+1)-Wgrid(jWstar));
    tPstar = (Pstar*Wstar - Wgrid(jPstar))/(Wgrid(jPstar+1)-Wgrid(jPstar));

    if A0 > ASmooth
        TotalIncome = Y0*(1-Deltay*D0) + (1+rP)*A0 - pE(jE) + YTransfer(iY);
    elseif A0 <= -ASmooth
        TotalIncome = Y0*(1-Deltay*D0) + (1+rM)*A0 - pE(jE) + YTransfer(iY);
    else
        r = rM + ((A0/ASmooth)/(1+(A0/ASmooth)^2)+0.5)*(rP-rM);
        TotalIncome = Y0*(1-Deltay*D0) + (1+r)*A0 - pE(jE) + YTransfer(iY);
    end
    
    w = Wstar - (1-delta)*W0;
    if w <= -wSmooth
        winvest = kappaZ;
    elseif w > wSmooth
        winvest = 1;
    else
        winvest = kappaZ + ((w/wSmooth)/(1+(w/wSmooth)^2)+0.5)*(1-kappaZ);
    end
    c = TotalIncome - w*winvest - Astar - pFactor * E0 * Wstar * Pstar^(1+phiP);
    
    if c <= lowC
        Astar = Agrid(1);
        jAstar = 0;
        for kA = 2:nA
            if Agrid(kA) > Astar
                jAstar = kA - 1;
                break
            elseif kA == nA
                jAstar = nA-1;
            end
        end
        
        Wstar = lowW + 2e-6;
        jWstar = 0;
        for kW = 2:nW
            if Wgrid(kW) > Wstar
                jWstar = kW - 1;
                break
            elseif kW == nW
                jWstar = nW-1;
            end
        end
        
        Pstar = 0.0;
        jPstar = 0;
        for kW = 2:nW
            if Wgrid(kW) > Pstar*Wstar
                jPstar = kW - 1;
                break
            elseif kW == nW
                jPstar = nW-1;
            end
        end
        tAstar = (Astar - Agrid(jAstar))/(Agrid(jAstar+1)-Agrid(jAstar));
        tWstar = (Wstar - Wgrid(jWstar))/(Wgrid(jWstar+1)-Wgrid(jWstar));
        tPstar = (Pstar*Wstar - Wgrid(jPstar))/(Wgrid(jPstar+1)-Wgrid(jPstar));
        
        w = Wstar - (1-delta)*W0;
        if w <= -wSmooth
            winvest = kappaZ;
        elseif w > wSmooth
            winvest = 1;
        else
            winvest = kappaZ + ((w/wSmooth)/(1+(w/wSmooth)^2)+0.5)*(1-kappaZ);
        end
        c = TotalIncome - w*winvest - Astar - pFactor * E0 * Wstar * Pstar^(1+phiP);
        
        if c < lowC + 1e-6
            GlobalStop = 1;
        else
            GlobalStop = 0;
            
        end
    else
        GlobalStop = 0;
    end
        
    if c <= lowC || Wstar <= lowW
        ValueStar = -1e9;
    else
        Utility1 = (((c-lowC)/scalefactor)^(1-gamma) - 1)/(1-gamma) + theta * ( ((Wstar-lowW)/scalefactor)^(1-beta) - 1)/(1-beta);
        V0C1 = 0;
        V0C2 = 0;
        for jY = 1:nY
            V0C1 = V0C1 + TransY(iY,jY) * ( (1-tAstar)*((1-tWstar)*v0(jE, jAstar  ,jWstar  ,jY,1)   ...
                                                        +  tWstar *v0(jE, jAstar  ,jWstar+1,jY,1))  ...
                                             + tAstar *((1-tWstar)*v0(jE, jAstar+1,jWstar  ,jY,1)   ...
                                                        +  tWstar *v0(jE, jAstar+1,jWstar+1,jY,1)));
            V0C2 = V0C2 + TransY(iY,jY) * ( (1-tAstar)*((1-tPstar)*v0(jE, jAstar  ,jPstar  ,jY,2)   ...
                                                        +  tPstar *v0(jE, jAstar  ,jPstar+1,jY,2))  ...
                                             + tAstar *((1-tPstar)*v0(jE, jAstar+1,jPstar  ,jY,2)   ...
                                                        +  tPstar *v0(jE, jAstar+1,jPstar+1,jY,2)));
        end
        ValueStar = Utility1 + rho2 * (a*E0*V0C2 + (1-a*E0)*V0C1);
    end
        
    iterFine = 0;
    stepsize = 1e-3;       
    
    while GlobalStop == 0 && iterFine < 1000
        %%%Calculate Gradients
        
        w = Wstar - (1-delta)*W0;
        if w <= -wSmooth
            winvest = kappaZ;
        elseif w > wSmooth
            winvest = 1;
        else
            winvest = kappaZ + ((w/wSmooth)/(1+(w/wSmooth)^2)+0.5)*(1-kappaZ);
        end
        c = TotalIncome - w*winvest - Astar - pFactor * E0 * Wstar * Pstar^(1+phiP);
        if c > lowC
            dUdc = ((c-lowC)/scalefactor)^(-gamma)/scalefactor;
        else
            dUdc = 1e6;
        end
        if Wstar > lowW
            dUdW = theta * ((Wstar-lowW)/scalefactor)^(-beta)/scalefactor;
        else
            dUdW = 1e6;
        end
        dVdAC1 = 0;
        dVdAC2 = 0;
        if tAstar <= 0.5 
            if jAstar == 1
                for jY = 1:nY
                    dVdAC1 = dVdAC1 + TransY(iY,jY) * ( ((1-tWstar)*v0(jE, jAstar+1,jWstar  ,jY,1) + tWstar*v0(jE, jAstar+1,jWstar+1,jY,1))  ...
                                                      - ((1-tWstar)*v0(jE, jAstar  ,jWstar  ,jY,1) + tWstar*v0(jE, jAstar  ,jWstar+1,jY,1)) );
                    
                    dVdAC2 = dVdAC2 + TransY(iY,jY) * (((1-tPstar)*v0(jE, jAstar+1,jPstar  ,jY,2) + tPstar*v0(jE, jAstar+1,jPstar+1,jY,2))  ...
                                                     - ((1-tPstar)*v0(jE, jAstar  ,jPstar  ,jY,2) + tPstar*v0(jE, jAstar  ,jPstar+1,jY,2)));
                end
                dVdA = ((1-a*E0)*dVdAC1 + a*E0*dVdAC2)/(Agrid(jAstar+1)-Agrid(jAstar));
            else
                tAstar2 = 0.5 + tAstar;
                for jY = 1:nY
                    dVdAC1 = dVdAC1 + TransY(iY,jY) * ( (1-tAstar2)*((1-tWstar)*v0(jE, jAstar  ,jWstar ,jY,1)   ...
                                                                     +  tWstar *v0(jE, jAstar  ,jWstar+1,jY,1))  ...
                                                          + tAstar2*((1-tWstar)*v0(jE, jAstar+1,jWstar  ,jY,1)   ...
                                                                     +  tWstar *v0(jE, jAstar+1,jWstar+1,jY,1)) - ...
                                                       ((1-tAstar2)*((1-tWstar)*v0(jE, jAstar-1,jWstar ,jY,1)   ...
                                                                     +  tWstar *v0(jE, jAstar-1,jWstar+1,jY,1))  ...
                                                          + tAstar2*((1-tWstar)*v0(jE, jAstar  ,jWstar  ,jY,1)   ...
                                                                     +  tWstar *v0(jE, jAstar  ,jWstar+1,jY,1))));
                    dVdAC2 = dVdAC2 + TransY(iY,jY) * ( (1-tAstar2)*((1-tPstar)*v0(jE, jAstar  ,jPstar  ,jY,2)   ...
                                                                     +  tPstar *v0(jE, jAstar  ,jPstar+1,jY,2))  ...
                                                          + tAstar2*((1-tPstar)*v0(jE, jAstar+1,jPstar  ,jY,2)   ...
                                                                     +  tPstar *v0(jE, jAstar+1,jPstar+1,jY,2)) - ...
                                                       ((1-tAstar2)*((1-tPstar)*v0(jE, jAstar-1,jPstar  ,jY,2)   ...
                                                                     +  tPstar *v0(jE, jAstar-1,jPstar+1,jY,2))  ...
                                                          + tAstar2*((1-tPstar)*v0(jE, jAstar  ,jPstar  ,jY,2)   ...
                                                                     +  tPstar *v0(jE, jAstar  ,jPstar+1,jY,2))));
                end
                dVdA = ((1-a*E0)*dVdAC1 + a*E0*dVdAC2)/(  ((1-tAstar2)*Agrid(jAstar)+tAstar2*Agrid(jAstar+1)) - ((1-tAstar2)*Agrid(jAstar-1)+tAstar2*Agrid(jAstar)) );
            end
        else
            if jAstar == nA-1
%                 for jY = 1:nY
%                     dVdAC1 = dVdAC1 + TransY(iY,jY) * (((1-tWstar)*v0(jE, jAstar+1,jWstar  ,jY,1) + tWstar*v0(jE, jAstar+1,jWstar+1,jY,1))  ...
%                                                      - ((1-tWstar)*v0(jE, jAstar  ,jWstar  ,jY,1) + tWstar*v0(jE, jAstar  ,jWstar+1,jY,1)));
%                     
%                     dVdAC2 = dVdAC2 + TransY(iY,jY) * (((1-tPstar)*v0(jE, jAstar+1,jPstar  ,jY,2) + tPstar*v0(jE, jAstar+1,jPstar+1,jY,2))  ...
%                                                      - ((1-tPstar)*v0(jE, jAstar  ,jPstar  ,jY,2) + tPstar*v0(jE, jAstar  ,jPstar+1,jY,2)));
%                 end
%                 dVdA = ((1-a*E0)*dVdAC1 + a*E0*dVdAC2)/(Agrid(jAstar+1)-Agrid(jAstar));
                tAstar2 = -0.5 + tAstar;
                for jY = 1:nY
                    dVdAC1 = dVdAC1 + TransY(iY,jY) * ( (1-tAstar2)*((1-tWstar)*v0(jE, jAstar+1,jWstar ,jY,1)   ...
                                                                     +  tWstar *v0(jE, jAstar+1,jWstar+1,jY,1))  ...
                                                          + tAstar2*((1-tWstar)*v0(jE, jAstar+1,jWstar  ,jY,1)   ...
                                                                     +  tWstar *v0(jE, jAstar+1,jWstar+1,jY,1)) - ...
                                                       ((1-tAstar2)*((1-tWstar)*v0(jE, jAstar  ,jWstar ,jY,1)   ...
                                                                     +  tWstar *v0(jE, jAstar  ,jWstar+1,jY,1))  ...
                                                          + tAstar2*((1-tWstar)*v0(jE, jAstar+1,jWstar  ,jY,1)   ...
                                                                     +  tWstar *v0(jE, jAstar+1,jWstar+1,jY,1))));
                    dVdAC2 = dVdAC2 + TransY(iY,jY) * ( (1-tAstar2)*((1-tPstar)*v0(jE, jAstar+1,jPstar  ,jY,2)   ...
                                                                     +  tPstar *v0(jE, jAstar+1,jPstar+1,jY,2))  ...
                                                          + tAstar2*((1-tPstar)*v0(jE, jAstar+1,jPstar  ,jY,2)   ...
                                                                     +  tPstar *v0(jE, jAstar+1,jPstar+1,jY,2)) - ...
                                                       ((1-tAstar2)*((1-tPstar)*v0(jE, jAstar  ,jPstar  ,jY,2)   ...
                                                                     +  tPstar *v0(jE, jAstar  ,jPstar+1,jY,2))  ...
                                                          + tAstar2*((1-tPstar)*v0(jE, jAstar+1,jPstar  ,jY,2)   ...
                                                                     +  tPstar *v0(jE, jAstar+1,jPstar+1,jY,2))));
                end
                dVdA = ((1-a*E0)*dVdAC1 + a*E0*dVdAC2)/(Agrid(jAstar+1)-Agrid(jAstar));
            
            else
                tAstar2 = -0.5 + tAstar;
                for jY = 1:nY
                    dVdAC1 = dVdAC1 + TransY(iY,jY) * ( (1-tAstar2)*((1-tWstar)*v0(jE, jAstar+1,jWstar ,jY,1)   ...
                                                                     +  tWstar *v0(jE, jAstar+1,jWstar+1,jY,1))  ...
                                                          + tAstar2*((1-tWstar)*v0(jE, jAstar+2,jWstar  ,jY,1)   ...
                                                                     +  tWstar *v0(jE, jAstar+2,jWstar+1,jY,1)) - ...
                                                       ((1-tAstar2)*((1-tWstar)*v0(jE, jAstar  ,jWstar ,jY,1)   ...
                                                                     +  tWstar *v0(jE, jAstar  ,jWstar+1,jY,1))  ...
                                                          + tAstar2*((1-tWstar)*v0(jE, jAstar+1,jWstar  ,jY,1)   ...
                                                                     +  tWstar *v0(jE, jAstar+1,jWstar+1,jY,1))));
                    dVdAC2 = dVdAC2 + TransY(iY,jY) * ( (1-tAstar2)*((1-tPstar)*v0(jE, jAstar+1,jPstar  ,jY,2)   ...
                                                                     +  tPstar *v0(jE, jAstar+1,jPstar+1,jY,2))  ...
                                                          + tAstar2*((1-tPstar)*v0(jE, jAstar+2,jPstar  ,jY,2)   ...
                                                                     +  tPstar *v0(jE, jAstar+2,jPstar+1,jY,2)) - ...
                                                       ((1-tAstar2)*((1-tPstar)*v0(jE, jAstar  ,jPstar  ,jY,2)   ...
                                                                     +  tPstar *v0(jE, jAstar  ,jPstar+1,jY,2))  ...
                                                          + tAstar2*((1-tPstar)*v0(jE, jAstar+1,jPstar  ,jY,2)   ...
                                                                     +  tPstar *v0(jE, jAstar+1,jPstar+1,jY,2))));
                end
                dVdA = ((1-a*E0)*dVdAC1 + a*E0*dVdAC2)/(  ((1-tAstar2)*Agrid(jAstar+1)+tAstar2*Agrid(jAstar+2)) - ((1-tAstar2)*Agrid(jAstar)+tAstar2*Agrid(jAstar+1)) );
            end 
        end
        GradA = rho2*dVdA - dUdc;
        
        dVdWC1 = 0;
        dVdWC2 = 0;
        if tWstar <= 0.5
            if jWstar == 1
                for jY = 1:nY
                    dVdWC1 = dVdWC1 + TransY(iY,jY) * (((1-tAstar)*v0(jE, jAstar,jWstar+1,jY,1) + tAstar*v0(jE, jAstar+1,jWstar+1,jY,1))  ...
                                                     - ((1-tAstar)*v0(jE, jAstar,jWstar  ,jY,1) + tAstar*v0(jE, jAstar+1,jWstar  ,jY,1)));
                end
                dVdWC1 = (1-a*E0)*dVdWC1 /(Wgrid(jWstar+1)-Wgrid(jWstar));
            else
                tWstar2 = 0.5 + tWstar;
                for jY = 1:nY
                    dVdWC1 = dVdWC1 + TransY(iY,jY) * ( (1-tAstar)*((1-tWstar2)*v0(jE, jAstar  ,jWstar  ,jY,1)   ...
                                                                    +  tWstar2 *v0(jE, jAstar  ,jWstar+1,jY,1))  ...
                                                          + tAstar*((1-tWstar2)*v0(jE, jAstar+1,jWstar  ,jY,1)   ...
                                                                     + tWstar2 *v0(jE, jAstar+1,jWstar+1,jY,1)) - ...
                                                       ((1-tAstar)*((1-tWstar2)*v0(jE, jAstar  ,jWstar-1,jY,1)   ...
                                                                    +  tWstar2 *v0(jE, jAstar  ,jWstar  ,jY,1))  ...
                                                          + tAstar*((1-tWstar2)*v0(jE, jAstar+1,jWstar-1,jY,1)   ...
                                                                    +  tWstar2 *v0(jE, jAstar+1,jWstar  ,jY,1))));
                end
                dVdWC1 = (1-a*E0)*dVdWC1 /(  ((1-tWstar2)*Wgrid(jWstar)+tWstar2*Wgrid(jWstar+1)) - ((1-tWstar2)*Wgrid(jWstar-1)+tWstar2*Wgrid(jWstar)) );
            end
        else
            if jWstar == nW - 1
%                 for jY = 1:nY
%                     dVdWC1 = dVdWC1 + TransY(iY,jY) * (((1-tAstar)*v0(jE, jAstar,jWstar+1,jY,1) + tAstar*v0(jE, jAstar+1,jWstar+1,jY,1))  ...
%                                                      - ((1-tAstar)*v0(jE, jAstar,jWstar  ,jY,1) + tAstar*v0(jE, jAstar+1,jWstar  ,jY,1)));
%                 end
                tWstar2 = -0.5 + tWstar;
                for jY = 1:nY
                    dVdWC1 = dVdWC1 + TransY(iY,jY) * ( (1-tAstar)*((1-tWstar2)*v0(jE, jAstar  ,jWstar+1,jY,1)   ...
                                                                    +  tWstar2 *v0(jE, jAstar  ,jWstar+1,jY,1))  ...
                                                        +   tAstar*((1-tWstar2)*v0(jE, jAstar+1,jWstar+1,jY,1)   ...
                                                                    +  tWstar2 *v0(jE, jAstar+1,jWstar+1,jY,1)) - ...
                                                       ((1-tAstar)*((1-tWstar2)*v0(jE, jAstar  ,jWstar  ,jY,1)   ...
                                                                    +  tWstar2 *v0(jE, jAstar  ,jWstar+1,jY,1))  ...
                                                          + tAstar*((1-tWstar2)*v0(jE, jAstar+1,jWstar  ,jY,1)   ...
                                                                    +  tWstar2 *v0(jE, jAstar+1,jWstar+1,jY,1))));
                end
                dVdWC1 = (1-a*E0)*dVdWC1 /(Wgrid(jWstar+1)-Wgrid(jWstar));
            else
                tWstar2 = -0.5 + tWstar;
                for jY = 1:nY
                    dVdWC1 = dVdWC1 + TransY(iY,jY) * ( (1-tAstar)*((1-tWstar2)*v0(jE, jAstar  ,jWstar+1,jY,1)   ...
                                                                    +  tWstar2 *v0(jE, jAstar  ,jWstar+2,jY,1))  ...
                                                        +   tAstar*((1-tWstar2)*v0(jE, jAstar+1,jWstar+1,jY,1)   ...
                                                                    +  tWstar2 *v0(jE, jAstar+1,jWstar+2,jY,1)) - ...
                                                       ((1-tAstar)*((1-tWstar2)*v0(jE, jAstar  ,jWstar  ,jY,1)   ...
                                                                    +  tWstar2 *v0(jE, jAstar  ,jWstar+1,jY,1))  ...
                                                          + tAstar*((1-tWstar2)*v0(jE, jAstar+1,jWstar  ,jY,1)   ...
                                                                    +  tWstar2 *v0(jE, jAstar+1,jWstar+1,jY,1))));
                end
                dVdWC1 = (1-a*E0)*dVdWC1 /(  ((1-tWstar2)*Wgrid(jWstar+1)+tWstar2*Wgrid(jWstar+2)) - ((1-tWstar2)*Wgrid(jWstar)+tWstar2*Wgrid(jWstar+1)) );
            end
        end
           
        if tPstar <= 0.5
            if jPstar == 1
                for jY = 1:nY
                    dVdWC2 = dVdWC2 + TransY(iY,jY) * (((1-tAstar)*v0(jE, jAstar,jPstar+1,jY,2) + tAstar*v0(jE, jAstar+1,jPstar+1,jY,2))  ...
                                                     - ((1-tAstar)*v0(jE, jAstar,jPstar  ,jY,2) + tAstar*v0(jE, jAstar+1,jPstar  ,jY,2)));
                end
                dVdWC2 = a*E0*dVdWC2 /(Wgrid(jPstar+1)-Wgrid(jPstar));
            else
                tPstar2 = 0.5 + tPstar;
                for jY = 1:nY
                    dVdWC2 = dVdWC2 + TransY(iY,jY) * ( (1-tAstar)*((1-tPstar2)*v0(jE, jAstar  ,jPstar  ,jY,2)   ...
                                                                    +  tPstar2 *v0(jE, jAstar  ,jPstar+1,jY,2))  ...
                                                       +    tAstar*((1-tPstar2)*v0(jE, jAstar+1,jPstar  ,jY,2)   ...
                                                                    +  tPstar2 *v0(jE, jAstar+1,jPstar+1,jY,2)) - ...
                                                       ((1-tAstar)*((1-tPstar2)*v0(jE, jAstar  ,jPstar-1,jY,2)   ...
                                                                    +  tPstar2 *v0(jE, jAstar  ,jPstar  ,jY,2))  ...
                                                          + tAstar*((1-tPstar2)*v0(jE, jAstar+1,jPstar-1,jY,2)   ...
                                                                    +  tPstar2 *v0(jE, jAstar+1,jPstar  ,jY,2))));
                end
                dVdWC2 = a*E0*dVdWC2 /(  ((1-tPstar2)*Wgrid(jPstar)+tPstar2*Wgrid(jPstar+1)) - ((1-tPstar2)*Wgrid(jPstar-1)+tPstar2*Wgrid(jPstar)) );
            end
        else
            if jPstar == nW - 1
%                 for jY = 1:nY
%                     dVdWC2 = dVdWC2 + TransY(iY,jY) * (((1-tAstar)*v0(jE, jAstar,jPstar+1,jY,2) + tAstar*v0(jE, jAstar+1,jPstar+1,jY,2))  ...
%                                                      - ((1-tAstar)*v0(jE, jAstar,jPstar  ,jY,2) + tAstar*v0(jE, jAstar+1,jPstar  ,jY,2)));
%                 end
                tPstar2 = -0.5 + tPstar;
                for jY = 1:nY
                    dVdWC2 = dVdWC2 + TransY(iY,jY) * ( (1-tAstar)*((1-tPstar2)*v0(jE, jAstar  ,jPstar+1,jY,2)   ...
                                                                    +  tPstar2 *v0(jE, jAstar  ,jPstar+1,jY,2))  ...
                                                       +    tAstar*((1-tPstar2)*v0(jE, jAstar+1,jPstar+1,jY,2)   ...
                                                                    +  tPstar2 *v0(jE, jAstar+1,jPstar+1,jY,2)) - ...
                                                       ((1-tAstar)*((1-tPstar2)*v0(jE, jAstar  ,jPstar  ,jY,2)   ...
                                                                    +  tPstar2 *v0(jE, jAstar  ,jPstar+1,jY,2))  ...
                                                          + tAstar*((1-tPstar2)*v0(jE, jAstar+1,jPstar  ,jY,2)   ...
                                                                    +  tPstar2 *v0(jE, jAstar+1,jPstar+1,jY,2))));
                end
                dVdWC2 = a*E0*dVdWC2 /(Wgrid(jPstar+1)-Wgrid(jPstar));
            else
                tPstar2 = -0.5 + tPstar;
                for jY = 1:nY
                    dVdWC2 = dVdWC2 + TransY(iY,jY) * ( (1-tAstar)*((1-tPstar2)*v0(jE, jAstar  ,jPstar+1,jY,2)   ...
                                                                    +  tPstar2 *v0(jE, jAstar  ,jPstar+2,jY,2))  ...
                                                       +    tAstar*((1-tPstar2)*v0(jE, jAstar+1,jPstar+1,jY,2)   ...
                                                                    +  tPstar2 *v0(jE, jAstar+1,jPstar+2,jY,2)) - ...
                                                       ((1-tAstar)*((1-tPstar2)*v0(jE, jAstar  ,jPstar  ,jY,2)   ...
                                                                    +  tPstar2 *v0(jE, jAstar  ,jPstar+1,jY,2))  ...
                                                          + tAstar*((1-tPstar2)*v0(jE, jAstar+1,jPstar  ,jY,2)   ...
                                                                    +  tPstar2 *v0(jE, jAstar+1,jPstar+1,jY,2))));
                end
                dVdWC2 = a*E0*dVdWC2 /(  ((1-tPstar2)*Wgrid(jPstar+1)+tPstar2*Wgrid(jPstar+2)) - ((1-tPstar2)*Wgrid(jPstar)+tPstar2*Wgrid(jPstar+1)) );
            end
        end
        
        dVdW = dVdWC1 + Pstar*a*E0*dVdWC2;
        GradW = rho2*dVdW - dUdc*(winvest + pFactor*E0*Pstar^(1+phiP)) + dUdW;
        
        dVdP = Wstar*a*E0*dVdWC2;
        GradP = rho2*dVdP - dUdc*pFactor*E0*Wstar*(1+phiP)*max(1e-9,Pstar)^phiP;
        
        %% Scale gradient to 1
        
        if Astar <= Agrid(1) && GradA < 0
            GradA = 0;
        end
        if Wstar <= lowW + 1e-6 && GradW < 0
            GradW = 0;
        end
        if (Pstar <= 0.0 && GradP < 0) || (Pstar >= 1.0 && GradP > 0.0)
            GradP = 0;
        end
        
        GradNorm = sqrt(GradA^2+GradW^2+GradP^2);
        GradAnew = GradA/GradNorm;
        GradWnew = GradW/GradNorm;
        GradPnew = GradP/GradNorm;
        
        if GradNorm > 1e-6
            %% Make step in Gradient direction
            LineStepStop = 0;
        
            while LineStepStop == 0
                
                Anew = Astar + stepsize*GradAnew;
                Anew = max(Agrid(1),min(Agrid(nA),Anew));
                Wnew = Wstar + stepsize*GradWnew;
                Wnew = max(lowW+1e-6,min(Wgrid(nW),Wnew));
                Pnew = Pstar + stepsize*GradPnew;
                Pnew = max(0.0,min(1.0,Pnew));
                
                jAnew = 0;
                for kA = 2:nA
                    if Agrid(kA) > Anew
                        jAnew = kA - 1;
                        break
                    elseif kA == nA
                        jAnew = nA-1;
                    end
                end
                
                jWnew = 0;
                for kW = 2:nW
                    if Wgrid(kW) > Wnew
                        jWnew = kW - 1;
                        break
                    elseif kW == nW
                        jWnew = nW-1;
                    end
                end
                
                jPnew = 0;
                for kW = 2:nW
                    if Wgrid(kW) > Pnew*Wnew
                        jPnew = kW - 1;
                        break
                    elseif kW == nW
                        jPnew = nW-1;
                    end
                end
                tAnew = (Anew - Agrid(jAnew))/(Agrid(jAnew+1)-Agrid(jAnew));
                tWnew = (Wnew - Wgrid(jWnew))/(Wgrid(jWnew+1)-Wgrid(jWnew));
                tPnew = (Pnew*Wnew - Wgrid(jPnew))/(Wgrid(jPnew+1)-Wgrid(jPnew));
                
                w = Wnew - (1-delta)*W0;
                if w <= -wSmooth
                    winvest = kappaZ;
                elseif w > wSmooth
                    winvest = 1;
                else
                    winvest = kappaZ + ((w/wSmooth)/(1+(w/wSmooth)^2)+0.5)*(1-kappaZ);
                end
                c = TotalIncome - w*winvest - Anew - pFactor * E0 * Wnew * Pnew^(1+phiP);
                
                if c <= lowC || Wnew <= lowW
                    ValueStarNew = -1e9;
                else
                    Utility1 = (((c-lowC)/scalefactor)^(1-gamma) - 1)/(1-gamma) + theta * ( ((Wnew-lowW)/scalefactor)^(1-beta) - 1)/(1-beta);
                    V0C1 = 0;
                    V0C2 = 0;
                    for jY = 1:nY
                        V0C1 = V0C1 + TransY(iY,jY) * ( (1-tAnew)*((1-tWnew)*v0(jE, jAnew  ,jWnew ,jY,1)   ...
                            +  tWnew *v0(jE, jAnew  ,jWnew+1,jY,1))  ...
                            + tAnew*((1-tWnew)*v0(jE, jAnew+1,jWnew  ,jY,1)   ...
                            +  tWnew *v0(jE, jAnew+1,jWnew+1,jY,1)));
                        V0C2 = V0C2 + TransY(iY,jY) * ( (1-tAnew)*((1-tPnew)*v0(jE, jAnew  ,jPnew ,jY,2)   ...
                            +  tPnew *v0(jE, jAnew  ,jPnew+1    ,jY,2))  ...
                            + tAnew*((1-tPnew)*v0(jE, jAnew+1,jPnew      ,jY,2)   ...
                            +  tPnew *v0(jE, jAnew+1,jPnew+1    ,jY,2)));
                    end
                    ValueStarNew = Utility1 + rho2 * (a*E0*V0C2 + (1-a*E0)*V0C1);
                end
                
                if ValueStarNew > ValueStar
                    LineStepStop = 1;
                    stepsize = min(1e-2,stepsize*1.25);
                    
                    Astar = Anew;
                    Wstar = Wnew;
                    Pstar = Pnew;
                    ValueStar = ValueStarNew;
                    jAstar = jAnew;
                    tAstar = tAnew;
                    jWstar = jWnew;
                    tWstar = tWnew;
                    jPstar = jPnew;
                    tPstar = tPnew;
%                    fprintf(" + | stepsize = %g | GradNorm = %g | ValueStar = %g \n",stepsize,GradNorm,ValueStar)
                elseif stepsize < 1e-9
                    LineStepStop = 1;
                    GlobalStop = 1;
                else
                    stepsize = stepsize*0.5;
 %                   fprintf(" - | stepsize = %g | GradNorm = %g | ValueStar = %g \n",stepsize,GradNorm,ValueStar)
                end
            end
            iterFine = iterFine + 1;
          
        else
            GlobalStop = 1;
        end        
    end
    PolA = Astar;
    PolW = Wstar;
    PolP = Pstar;
    Value = ValueStar;
end


function [v1Mov,PolE,PolA,PolW,PolP] = ValueCompareMov(iA,iW,iY,iD)
   PolE = Egrid(nE);
   PolA = Agrid(1);
   PolW = lowW + 1e-6;
   PolP = 0.0;
   v1Mov = -1e9;
   
   for ii = 1:nE
       if v1MovHelp(ii,iA,iW,iY,iD) > v1Mov
           v1Mov = v1MovHelp(ii,iA,iW,iY,iD);
           PolA = PolAMovHelp(ii,iA,iW,iY,iD);
           PolW = PolWMovHelp(ii,iA,iW,iY,iD);
           PolP = PolPMovHelp(ii,iA,iW,iY,iD);
           PolE = Egrid(ii);
       end
   end
end

function [v1Fine,PolE,PolA,PolW,PolP,PolI] = ValueCompareFine(iE,iA,iW,iY,iD)
    if v1MovFine(iA,iW,iY,iD) > v1StayFine(iE,iA,iW,iY,iD)
        if PolicyMovFineE(iA,iW,iY,iD) == Egrid(iE)
            PolI = 0;
        else
            PolI = 1;
        end
        v1Fine = v1MovFine(iA,iW,iY,iD);
        PolE = PolicyMovFineE(iA,iW,iY,iD);
        PolA = PolicyMovFineA(iA,iW,iY,iD);
        PolW = PolicyMovFineW(iA,iW,iY,iD);
        PolP = PolicyMovFineP(iA,iW,iY,iD);
    else
        v1Fine = v1StayFine(iE,iA,iW,iY,iD);
        PolE = Egrid(iE);
        PolA = PolicyStayFineA(iE,iA,iW,iY,iD);
        PolW = PolicyStayFineW(iE,iA,iW,iY,iD);
        PolP = PolicyStayFineP(iE,iA,iW,iY,iD);
        PolI = 0;
    end
end

% ---------------------------------------------------------------------------------------------------------------------
%   Fine Policy Iteration
%-----------------------------------------------------------------------------------------------------------------------

function v2Fine = PolicyIterationMovFine(iEstar,iA,iW,iY,iD) 
    
    A0 = Agrid(iA);
    W0 = Wgrid(iW);
    Y0 = Ygrid(iY);
    D0 = Dgrid(iD);
        
    Estar = Egrid(iEstar);
    Astar = PolAMovHelp(iEstar,iA,iW,iY,iD);
    Wstar = PolWMovHelp(iEstar,iA,iW,iY,iD);
    Pstar = PolPMovHelp(iEstar,iA,iW,iY,iD);
    
    jE = iEstar;
    
    jP = 0;
    for kW = 2:nW
        if Wgrid(kW) > Pstar*Wstar
            jP = kW - 1;
            break
        elseif kW == nW
            jP = nW-1;
        end
    end
        
    jA = 0;
    for kA = 2:nA
        if Agrid(kA) > Astar
            jA = kA - 1;
            break
        elseif kA == nA
            jA = nA-1;
        end
    end
    
    jW = 0;
    for kW = 2:nW
        if Wgrid(kW) > Wstar
            jW = kW - 1;
            break
        elseif kW == nW
            jW = nW-1;
        end
    end
    
    tA = (Astar - Agrid(jA))/(Agrid(jA+1)-Agrid(jA));
    tW = (Wstar - Wgrid(jW))/(Wgrid(jW+1)-Wgrid(jW));
    tP = (Pstar*Wstar - Wgrid(jP))/(Wgrid(jP+1)-Wgrid(jP));
       
    w = Wstar - (1-delta)*(1-DeltaW)*W0;
    if w <= -wSmooth
        winvest = kappaZ;
    elseif w > wSmooth
        winvest = 1;
    else
        winvest = kappaZ + ((w/wSmooth)/(1+(w/wSmooth)^2)+0.5)*(1-kappaZ);
    end
    w2 = w*winvest;
    
    if Wstar > lowW
        UWPart2 = theta * ((( Wstar-lowW )/scalefactor)^(1-beta)-1)/(1-beta);
    else
        UWPart2 = -1e6;
    end
    
    if A0 > ASmooth
        c = Y0*(1-Deltay*D0) + YTransfer(iY) + (1+rP)*A0 - Astar - w2 - pE(jE) - ...
                        pFactor * Estar * Wstar * Pstar^(1+phiP);
    elseif A0 <= -ASmooth
        c = Y0*(1-Deltay*D0) + YTransfer(iY) + (1+rM)*A0 - Astar - w2 - pE(jE) - ...
                        pFactor * Estar * Wstar * Pstar^(1+phiP);
    else
        r = rM + ((A0/ASmooth)/(1+(A0/ASmooth)^2)+0.5)*(rP-rM);
        c = Y0*(1-Deltay*D0) + YTransfer(iY) + (1+r)*A0 - Astar - w2 - pE(jE) - ...
                        pFactor * Estar * Wstar * Pstar^(1+phiP);
    end
           
    
    if c <= lowC || Wstar <= lowW
        v2Fine = -1e9;
    else        
        V0C1 = 0;
        V0C2 = 0;
        for jY = 1:nY
            V0C1 = V0C1 + TransY(iY,jY) * ( (1-tA)*((1-tW)*v1(jE, jA  ,jW  ,jY,1)   ...
                                                    +  tW *v1(jE, jA  ,jW+1,jY,1))  ...
                                             + tA *((1-tW)*v1(jE, jA+1,jW  ,jY,1)   ...
                                                    +  tW *v1(jE, jA+1,jW+1,jY,1)));
            V0C2 = V0C2 + TransY(iY,jY) * ( (1-tA)*((1-tP)*v1(jE, jA  ,jP  ,jY,2)   ...
                                                    +  tP *v1(jE, jA  ,jP+1,jY,2))  ...
                                             + tA *((1-tP)*v1(jE, jA+1,jP  ,jY,2)   ...
                                                    +  tP *v1(jE, jA+1,jP+1,jY,2)));
        end
        v2Fine = (((c-lowC)/scalefactor)^(1-gamma) - 1)/(1-gamma) + UWPart2 + rho2 * (a*Estar*V0C2 + (1-a*Estar)*V0C1);
    end
end

function [v2Fine,PolE,PolA,PolW,PolP] = ValueCompareMovPI(iA,iW,iY,iD)
   jEmax = 1;
   Vbest = v2MovHelp(1,iA,iW,iY,iD);
   
   for ii = 2:nE
       if v2MovHelp(ii,iA,iW,iY,iD) > Vbest
           jEmax = ii;
           Vbest = v2MovHelp(ii,iA,iW,iY,iD);
       end
   end
   
   v2Fine = v2MovHelp(jEmax,iA,iW,iY,iD);
   PolE = Egrid(jEmax);
   PolA = PolAMovHelp(jEmax,iA,iW,iY,iD);
   PolW = PolWMovHelp(jEmax,iA,iW,iY,iD);
   PolP = PolPMovHelp(jEmax,iA,iW,iY,iD);
end

function v2Fine = PolicyIterationStayFine(iE,iA,iW,iY,iD) 
    
    A0 = Agrid(iA);
    W0 = Wgrid(iW);
    Y0 = Ygrid(iY);
    D0 = Dgrid(iD);
    
    DeltaWHelp = 0;
    
    jEstar = iE;
    Estar = Egrid(jEstar);
    Astar = PolicyStayFineA(iE,iA,iW,iY,iD);
    Wstar = PolicyStayFineW(iE,iA,iW,iY,iD);
    Pstar = PolicyStayFineP(iE,iA,iW,iY,iD);
    
    jP = 0;
    for kW = 2:nW
        if Wgrid(kW) > Pstar*Wstar
            jP = kW - 1;
            break
        elseif kW == nW
            jP = nW-1;
        end
    end
    
    jA = 0;
    for kA = 2:nA
        if Agrid(kA) > Astar
            jA = kA - 1;
            break
        elseif kA == nA
            jA = nA-1;
        end
    end
    
    jW = 0;
    for kW = 2:nW
        if Wgrid(kW) > Wstar
            jW = kW - 1;
            break
        elseif kW == nW
            jW = nW-1;
        end
    end
    if jW == 0
        jW = nW-1;
    end
    
    tA = (Astar - Agrid(jA))/(Agrid(jA+1)-Agrid(jA));
    tW = (Wstar - Wgrid(jW))/(Wgrid(jW+1)-Wgrid(jW));
    tP = (Pstar*Wstar - Wgrid(jP))/(Wgrid(jP+1)-Wgrid(jP));
       
    w = Wstar - (1-delta)*(1-DeltaWHelp)*W0;
    if w <= -wSmooth
        winvest = kappaZ;
    elseif w > wSmooth
        winvest = 1;
    else
        winvest = kappaZ + ((w/wSmooth)/(1+(w/wSmooth)^2)+0.5)*(1-kappaZ);
    end
    w2 = w*winvest;
    
    if Wstar > lowW
        UWPart2 = theta * ((( Wstar-lowW )/scalefactor)^(1-beta)-1)/(1-beta);
    else
        UWPart2 = -1e6;
    end
    
    if A0 > ASmooth
        c = Y0*(1-Deltay*D0) + YTransfer(iY) + (1+rP)*A0 - Astar - w2 - pE(iE) - ...
                        pFactor * Estar * Wstar * Pstar^(1+phiP);
    elseif A0 <= -ASmooth
        c = Y0*(1-Deltay*D0) + YTransfer(iY) + (1+rM)*A0 - Astar - w2 - pE(iE) - ...
                        pFactor * Estar * Wstar * Pstar^(1+phiP);
    else
        r = rM + ((A0/ASmooth)/(1+(A0/ASmooth)^2)+0.5)*(rP-rM);
        c = Y0*(1-Deltay*D0) + YTransfer(iY) + (1+r)*A0 - Astar - w2 - pE(iE) - ...
                        pFactor * Estar * Wstar * Pstar^(1+phiP);
    end
    
    
    
    
    if c <= lowC || Wstar <= lowW
        v2Fine = -1e9;
    else        
        V0C1 = 0;
        V0C2 = 0;
        for jY = 1:nY
            V0C1 = V0C1 + TransY(iY,jY) * ( (1-tA)*((1-tW)*v1(jEstar, jA  ,jW  ,jY,1)   ...
                                                    +  tW *v1(jEstar, jA  ,jW+1,jY,1))  ...
                                             + tA *((1-tW)*v1(jEstar, jA+1,jW  ,jY,1)   ...
                                                    +  tW *v1(jEstar, jA+1,jW+1,jY,1)));
            V0C2 = V0C2 + TransY(iY,jY) * ( (1-tA)*((1-tP)*v1(jEstar, jA  ,jP  ,jY,2)   ...
                                                    +  tP *v1(jEstar, jA  ,jP+1,jY,2))  ...
                                             + tA *((1-tP)*v1(jEstar, jA+1,jP  ,jY,2)   ...
                                                    +  tP *v1(jEstar, jA+1,jP+1,jY,2)));
        end
        v2Fine = (((c-lowC)/scalefactor)^(1-gamma) - 1)/(1-gamma) + UWPart2 + rho2 * (a*Estar*V0C2 + (1-a*Estar)*V0C1);
    end
end

function [v2Fine,PolE,PolA,PolW,PolP,PolI] = ValueComparePIFine(iE,iA,iW,iY,iD)
    if v2MovFine(iA,iW,iY,iD) > v2StayFine(iE,iA,iW,iY,iD)
        if PolicyMovFineE(iA,iW,iY,iD) == Egrid(iE)
            PolI = 0;
        else
            PolI = 1;
        end
        v2Fine = v2MovFine(iA,iW,iY,iD);
        PolE = PolicyMovFineE(iA,iW,iY,iD);
        PolA = PolicyMovFineA(iA,iW,iY,iD);
        PolW = PolicyMovFineW(iA,iW,iY,iD);
        PolP = PolicyMovFineP(iA,iW,iY,iD);
    else
        v2Fine = v2StayFine(iE,iA,iW,iY,iD);
        PolE = Egrid(iE);
        PolA = PolicyStayFineA(iE,iA,iW,iY,iD);
        PolW = PolicyStayFineW(iE,iA,iW,iY,iD);
        PolP = PolicyStayFineP(iE,iA,iW,iY,iD);
        PolI = 0;
    end
end


%% ---------------------------------------------------------------------------------------------------------------------
%   Additional variables
%-----------------------------------------------------------------------------------------------------------------------

function [w,c,Vul] = AdditionalVariablesFine(iE,iA,iW,iY,iD)
    
    A0 = Agrid(iA);
    W0 = Wgrid(iW);
    Y0 = Ygrid(iY);
    D0 = Dgrid(iD);
    
    if PolicyFineI(iE,iA,iW,iY,iD) == 1
        DeltaWHelp = DeltaW;
    else
        DeltaWHelp = 0;
    end
    
    Estar = PolicyFineE(iE,iA,iW,iY,iD);
    jEstar = 0;
    for kE = 1:nE
        if Egrid(kE) >= PolicyFineE(iE,iA,iW,iY,iD)
            jEstar = kE;
            break
        end
    end
        
    Astar = PolicyFineA(iE,iA,iW,iY,iD);
    Wstar = PolicyFineW(iE,iA,iW,iY,iD);
    Pstar = PolicyFineP(iE,iA,iW,iY,iD);
    
    jP = 0;
    for kW = 2:nW
        if Wgrid(kW) > Pstar*Wstar
            jP = kW - 1;
            break
        elseif kW == nW
            jP = nW-1;
        end
    end    
    jA = 0;
    for kA = 2:nA
        if Agrid(kA) > Astar
            jA = kA - 1;
            break
        elseif kA == nA
            jA = nA - 1;
        end
    end
    jW = 0;
    for kW = 2:nW
        if Wgrid(kW) > Wstar
            jW = kW - 1;
            break
        elseif kW == nW
            jW = nW-1;
        end
    end
    
    tA = (Astar - Agrid(jA))/(Agrid(jA+1)-Agrid(jA));
    tW = (Wstar - Wgrid(jW))/(Wgrid(jW+1)-Wgrid(jW));
    tP = (Pstar*Wstar - Wgrid(jP))/(Wgrid(jP+1)-Wgrid(jP));
       
    w = Wstar - (1-delta)*(1-DeltaWHelp)*W0;
    if w <= -wSmooth
        winvest = kappaZ;
    elseif w > wSmooth
        winvest = 1;
    else
        winvest = kappaZ + ((w/wSmooth)/(1+(w/wSmooth)^2)+0.5)*(1-kappaZ);
    end
    w2 = w*winvest;

    if A0 > ASmooth
        c = Y0*(1-Deltay*D0) + YTransfer(iY) + (1+rP)*A0 - Astar - w2 - pE(jEstar) - pFactor * Estar * Wstar * Pstar^(1+phiP);
    elseif A0 <= -ASmooth
        c = Y0*(1-Deltay*D0) + YTransfer(iY) + (1+rM)*A0 - Astar - w2 - pE(jEstar) - pFactor * Estar * Wstar * Pstar^(1+phiP);
    else
        r = rM + ((A0/ASmooth)/(1+(A0/ASmooth)^2)+0.5)*(rP-rM);
        c = Y0*(1-Deltay*D0) + YTransfer(iY) + (1+r)*A0 - Astar - w2 - pE(jEstar) - pFactor * Estar * Wstar * Pstar^(1+phiP);
    end
    
    
    
    ExpU1 = 0;
    ExpU2 = 0;
    for jY = 1:nY
        ExpU1 = ExpU1 + TransY(iY,jY) * ( (1-tA)*((1-tW)*v0(jEstar, jA  ,jW  ,jY,1)   ...
                                                  +  tW *v0(jEstar, jA  ,jW+1,jY,1))  ...
                                           + tA *((1-tW)*v0(jEstar, jA+1,jW  ,jY,1)   ...
                                                  +  tW *v0(jEstar, jA+1,jW+1,jY,1)));
        ExpU2 = ExpU2 + TransY(iY,jY) * ( (1-tA)*((1-tP)*v0(jEstar, jA  ,jP  ,jY,2)   ...
                                                  +  tP *v0(jEstar, jA  ,jP+1,jY,2))  ...
                                           + tA *((1-tP)*v0(jEstar, jA+1,jP  ,jY,2)   ...
                                                  +  tP *v0(jEstar, jA+1,jP+1,jY,2)));
    end
    Vul = (ExpU1 - ExpU2)/(ExpU1-minV0);
end

function [w,c,Vul] = AdditionalVariablesMovFine(iA,iW,iY,iD)
    
    A0 = Agrid(iA);
    W0 = Wgrid(iW);
    Y0 = Ygrid(iY);
    D0 = Dgrid(iD);
    
    Estar = PolicyMovFineE(iA,iW,iY,iD);
    jEstar = 0;
    for kE = 1:nE
        if Egrid(kE) >= Estar
            jEstar = kE;
            break
        end
    end   
           
    Astar = PolicyMovFineA(iA,iW,iY,iD);
    Wstar = PolicyMovFineW(iA,iW,iY,iD);
    Pstar = PolicyMovFineP(iA,iW,iY,iD);
    
    jP = 0;
    for kW = 2:nW
        if Wgrid(kW) > Pstar*Wstar
            jP = kW - 1;
            break
        elseif kW == nW
            jP = nW-1;
        end
    end    
    jA = 0;
    for kA = 2:nA
        if Agrid(kA) > Astar
            jA = kA - 1;
            break
        elseif kA == nA
            jA = nA - 1;
        end
    end
    jW = 0;
    for kW = 2:nW
        if Wgrid(kW) > Wstar
            jW = kW - 1;
            break
        elseif kW == nW
            jW = nW-1;
        end
    end
        
    tA = (Astar - Agrid(jA))/(Agrid(jA+1)-Agrid(jA));
    tW = (Wstar - Wgrid(jW))/(Wgrid(jW+1)-Wgrid(jW));
    tP = (Pstar*Wstar - Wgrid(jP))/(Wgrid(jP+1)-Wgrid(jP));
       
    w = Wstar - (1-delta)*(1-DeltaW)*W0;
    if w <= -wSmooth
        winvest = kappaZ;
    elseif w > wSmooth
        winvest = 1;
    else
        winvest = kappaZ + ((w/wSmooth)/(1+(w/wSmooth)^2)+0.5)*(1-kappaZ);
    end
    w2 = w*winvest;
    
    if A0 > ASmooth
        c = Y0*(1-Deltay*D0) + YTransfer(iY) + (1+rP)*A0 - Astar - w2 - pE(jEstar) - pFactor * Estar * Wstar * Pstar^(1+phiP);
    elseif A0 <= -ASmooth
        c = Y0*(1-Deltay*D0) + YTransfer(iY) + (1+rM)*A0 - Astar - w2 - pE(jEstar) - pFactor * Estar * Wstar * Pstar^(1+phiP);        
    else
        r = rM + ((A0/ASmooth)/(1+(A0/ASmooth)^2)+0.5)*(rP-rM);
        c = Y0*(1-Deltay*D0) + YTransfer(iY) + (1+r)*A0 - Astar - w2 - pE(jEstar) - pFactor * Estar * Wstar * Pstar^(1+phiP);        
    end
    ExpU1 = 0;
    ExpU2 = 0;
    for jY = 1:nY
        ExpU1 = ExpU1 + TransY(iY,jY) * ( (1-tA)*((1-tW)*v0(jEstar, jA  ,jW  ,jY,1)   ...
                                                  +  tW *v0(jEstar, jA  ,jW+1,jY,1))  ...
                                           + tA *((1-tW)*v0(jEstar, jA+1,jW  ,jY,1)   ...
                                                  +  tW *v0(jEstar, jA+1,jW+1,jY,1)));
        ExpU2 = ExpU2 + TransY(iY,jY) * ( (1-tA)*((1-tP)*v0(jEstar, jA  ,jP  ,jY,2)   ...
                                                  +  tP *v0(jEstar, jA  ,jP+1,jY,2))  ...
                                           + tA *((1-tP)*v0(jEstar, jA+1,jP  ,jY,2)   ...
                                                  +  tP *v0(jEstar, jA+1,jP+1,jY,2)));
    end
    Vul = (ExpU1 - ExpU2)/(ExpU1-minV0);
end

function [w,c,Vul] = AdditionalVariablesStayFine(iE,iA,iW,iY,iD)
    
    A0 = Agrid(iA);
    W0 = Wgrid(iW);
    Y0 = Ygrid(iY);
    D0 = Dgrid(iD);
    
    jEstar = iE;
    Estar = Egrid(jEstar);   
           
    Astar = PolicyStayFineA(iE,iA,iW,iY,iD);
    Wstar = PolicyStayFineW(iE,iA,iW,iY,iD);
    Pstar = PolicyStayFineP(iE,iA,iW,iY,iD);
    
    jP = 0;
    for kW = 2:nW
        if Wgrid(kW) > Pstar*Wstar
            jP = kW - 1;
            break
        elseif kW == nW
            jP = nW-1;
        end
    end    
    jA = 0;
    for kA = 2:nA
        if Agrid(kA) > Astar
            jA = kA - 1;
            break
        elseif kA == nA
            jA = nA - 1;
        end
    end
    jW = 0;
    for kW = 2:nW
        if Wgrid(kW) > Wstar
            jW = kW - 1;
            break
        elseif kW == nW
            jW = nW-1;
        end
    end
    
    tA = (Astar - Agrid(jA))/(Agrid(jA+1)-Agrid(jA));
    tW = (Wstar - Wgrid(jW))/(Wgrid(jW+1)-Wgrid(jW));
    tP = (Pstar*Wstar - Wgrid(jP))/(Wgrid(jP+1)-Wgrid(jP));
       
    w = Wstar - (1-delta)*W0;
    if w <= -wSmooth
        winvest = kappaZ;
    elseif w > wSmooth
        winvest = 1;
    else
        winvest = kappaZ + ((w/wSmooth)/(1+(w/wSmooth)^2)+0.5)*(1-kappaZ);
    end
    w2 = w*winvest;
    
    if A0 > ASmooth
        c = Y0*(1-Deltay*D0) + YTransfer(iY) + (1+rP)*A0 - Astar - w2 - pE(jEstar) - pFactor * Estar * Wstar * Pstar^(1+phiP);
    elseif A0 <= -ASmooth
        c = Y0*(1-Deltay*D0) + YTransfer(iY) + (1+rM)*A0 - Astar - w2 - pE(jEstar) - pFactor * Estar * Wstar * Pstar^(1+phiP);
    else
        r = rM + ((A0/ASmooth)/(1+(A0/ASmooth)^2)+0.5)*(rP-rM);
        c = Y0*(1-Deltay*D0) + YTransfer(iY) + (1+r)*A0 - Astar - w2 - pE(jEstar) - pFactor * Estar * Wstar * Pstar^(1+phiP);
    end
            
    ExpU1 = 0;
    ExpU2 = 0;
    for jY = 1:nY
        ExpU1 = ExpU1 + TransY(iY,jY) * ( (1-tA)*((1-tW)*v0(jEstar, jA  ,jW  ,jY,1)   ...
                                                  +  tW *v0(jEstar, jA  ,jW+1,jY,1))  ...
                                           + tA *((1-tW)*v0(jEstar, jA+1,jW  ,jY,1)   ...
                                                  +  tW *v0(jEstar, jA+1,jW+1,jY,1)));
        ExpU2 = ExpU2 + TransY(iY,jY) * ( (1-tA)*((1-tP)*v0(jEstar, jA  ,jP  ,jY,2)   ...
                                                  +  tP *v0(jEstar, jA  ,jP+1,jY,2))  ...
                                           + tA *((1-tP)*v0(jEstar, jA+1,jP  ,jY,2)   ...
                                                  +  tP *v0(jEstar, jA+1,jP+1,jY,2)));
    end
    Vul = (ExpU1 - ExpU2)/(ExpU1-minV0);
end



function [MonotonicityA,MonotonicityW,MonotonicityY,MonotonicityD] = TestMonotonicityV0(iE,iA,iW,iY,iD)
    MonotonicityA = 1;
    MonotonicityW = 1;
    MonotonicityY = 1;
    MonotonicityD = 1;
    
    if v0(iE,iA,iW,iY,iD) - v0(iE,max(iA-1,1),iW,iY,iD) < 0
        MonotonicityA = 0;
    end
    if v0(iE,iA,iW,iY,iD) - v0(iE,iA,max(iW-1,1),iY,iD) < 0
        MonotonicityW = 0;
    end
    if v0(iE,iA,iW,iY,iD) - v0(iE,iA,iW,max(1,iY-1),iD) < 0
        MonotonicityY = 0;
    end
    if v0(iE,iA,iW,iY,1) < v0(iE,iA,iW,iY,2)
        MonotonicityD = 0;
    end
end

function [MonotonicityA,MonotonicityW,MonotonicityY,MonotonicityD] = TestMonotonicityV1(iE,iA,iW,iY,iD)
    MonotonicityA = 1;
    MonotonicityW = 1;
    MonotonicityY = 1;
    MonotonicityD = 1;
    
    if v1(iE,iA,iW,iY,iD) - v1(iE,max(iA-1,1),iW,iY,iD) < 0
        MonotonicityA = 0;
    end
    if v1(iE,iA,iW,iY,iD) - v1(iE,iA,max(iW-1,1),iY,iD) < 0
        MonotonicityW = 0;
    end
    if v1(iE,iA,iW,iY,iD) - v1(iE,iA,iW,max(1,iY-1),iD) < 0
        MonotonicityY = 0;
    end
    if v1(iE,iA,iW,iY,1) < v1(iE,iA,iW,iY,2)
        MonotonicityD = 0;
    end
end

function [MonotonicityA,MonotonicityW,MonotonicityY,MonotonicityD] = TestMonotonicityV2(iE,iA,iW,iY,iD)
    MonotonicityA = 1;
    MonotonicityW = 1;
    MonotonicityY = 1;
    MonotonicityD = 1;
    
    if v2(iE,iA,iW,iY,iD) - v2(iE,max(iA-1,1),iW,iY,iD) < 0
        MonotonicityA = 0;
    end
    if v2(iE,iA,iW,iY,iD) - v2(iE,iA,max(iW-1,1),iY,iD) < 0
        MonotonicityW = 0;
    end
    if v2(iE,iA,iW,iY,iD) - v2(iE,iA,iW,max(1,iY-1),iD) < 0
        MonotonicityY = 0;
    end
    if v2(iE,iA,iW,iY,1) < v2(iE,iA,iW,iY,2)
        MonotonicityD = 0;
    end
end


function v = MonotoniceA(v)
    for iE = 1:nE
        for iD = 1:nD
            for iY = 1:nY
                for iW = 1:nW
                    DiffA = diff(v(iE,:,iW,iY,iD));
                    if any(DiffA<0)
                        iA = 1;
                        while iA < nA
                            if DiffA(iA) < 0
                                jA = 2;
                                while v(iE,iA,iW,iY,iD) > v(iE,iA+jA,iW,iY,iD) && iA + jA < nA
                                    jA = jA+1;
                                end
                                v(iE,iA:(iA+jA),iW,iY,iD) = v(iE,iA,iW,iY,iD) + (v(iE,iA+jA,iW,iY,iD)-v(iE,iA,iW,iY,iD))/(Agrid(iA+jA)-Agrid(iA)) * (Agrid(iA:(iA+jA))-Agrid(iA));
                                iA = iA + jA + 1;
                            else
                                iA = iA + 1;
                            end
                        end                        
                    end
                end
            end
        end
    end    
end

function v = MonotoniceW(v)
    for iE = 1:nE
        for iD = 1:nD
            for iY = 1:nY
                for iA = 1:nA
                    DiffW = diff(v(iE,iA,:,iY,iD));
                    if any(DiffW<0)
                        iW = 1;
                        while iW < nW
                            if DiffW(iW) < 0
                                jW = 2;
                                while v(iE,iA,iW,iY,iD) > v(iE,iA,iW+jW,iY,iD) && iW + jW < nW
                                    jW = jW+1;
                                end
                                v(iE,iA,iW:(iW+jW),iY,iD) = v(iE,iA,iW,iY,iD) + (v(iE,iA,iW+jW,iY,iD)-v(iE,iA,iW,iY,iD))/(Wgrid(iW+jW)-Wgrid(iW)) * (Wgrid(iW:(iW+jW))-Wgrid(iW));
                                iW = iW + jW + 1;
                            else
                                iW = iW + 1;
                            end
                        end                        
                    end
                end
            end
        end
    end    
end


function [v1Fine,Estar,Astar,Wstar,Pstar,Istar,...
                       AstarMov,WstarMov,PstarMov,...
                       AstarStay,WstarStay,PstarStay] = MonotoniceV1_A(iE,iA,iW,iY,iD)
    
    Estar = PolicyFineE(iE,iA,iW,iY,iD);
    Astar = PolicyFineA(iE,iA,iW,iY,iD);
    Wstar = PolicyFineW(iE,iA,iW,iY,iD);
    Pstar = PolicyFineP(iE,iA,iW,iY,iD);
    Istar = PolicyFineI(iE,iA,iW,iY,iD);
    v1Fine = v1(iE,iA,iW,iY,iD);
    AstarMov = PolAMovHelp(iE,iA,iW,iY,iD);
    WstarMov = PolWMovHelp(iE,iA,iW,iY,iD);
    PstarMov = PolPMovHelp(iE,iA,iW,iY,iD);
    AstarStay = PolicyStayFineA(iE,iA,iW,iY,iD);
    WstarStay = PolicyStayFineW(iE,iA,iW,iY,iD);
    PstarStay = PolicyStayFineP(iE,iA,iW,iY,iD);   
    
    if MonotonicityA(iE,iA,iW,iY,iD) == 0
        A0 = Agrid(iA);
        W0 = Wgrid(iW);
        Y0 = Ygrid(iY);
        D0 = Dgrid(iD);
        
        Estar = PolicyFineE(iE,iA-1,iW,iY,iD);
        Astar = PolicyFineA(iE,iA-1,iW,iY,iD);
        Wstar = PolicyFineW(iE,iA-1,iW,iY,iD);
        Pstar = PolicyFineP(iE,iA-1,iW,iY,iD);
        Istar = PolicyFineI(iE,iA-1,iW,iY,iD);
        if Istar == 1
            AstarMov = Astar;
            WstarMov = Wstar;
            PstarMov = Pstar;
        else
            AstarStay = Astar;
            WstarStay = Wstar;
            PstarStay = Pstar;
        end
    
        jE = 0;
        for kE = 2:nE
            if Egrid(kE) > Estar
                jE = kE - 1;
                break
            elseif kE == nE
                jE = nE-1;
            end
        end
        
        jP = 0;
        for kW = 2:nW
            if Wgrid(kW) > Pstar*Wstar
                jP = kW - 1;
                break
            elseif kW == nW
                jP = nW-1;
            end
        end

        jA = 0;
        for kA = 2:nA
            if Agrid(kA) > Astar
                jA = kA - 1;
                break
            elseif kA == nA
                jA = nA-1;
            end
        end
    
        jW = 0;
        for kW = 2:nW
            if Wgrid(kW) > Wstar
                jW = kW - 1;
                break
            elseif kW == nW
                jW = nW-1;
            end
        end

        tA = (Astar - Agrid(jA))/(Agrid(jA+1)-Agrid(jA));
        tW = (Wstar - Wgrid(jW))/(Wgrid(jW+1)-Wgrid(jW));
        tP = (Pstar*Wstar - Wgrid(jP))/(Wgrid(jP+1)-Wgrid(jP));
       
        w = Wstar - (1-delta)*(1-DeltaW*Istar)*W0;
        if w <= -wSmooth
            winvest = kappaZ;
        elseif w > wSmooth
            winvest = 1;
        else
            winvest = kappaZ + ((w/wSmooth)/(1+(w/wSmooth)^2)+0.5)*(1-kappaZ);
        end
        w2 = w*winvest;
    
        if Wstar > lowW
            UWPart2 = theta * ((( Wstar-lowW )/scalefactor)^(1-beta)-1)/(1-beta);
        else
            UWPart2 = -1e6;
        end
    
        if A0 > ASmooth
            c = Y0*(1-Deltay*D0) + YTransfer(iY) + (1+rP)*A0 - Astar - w2 - pE(jE) - ...
                            pFactor * Estar * Wstar * Pstar^(1+phiP);
        elseif A0 <= -ASmooth
            c = Y0*(1-Deltay*D0) + YTransfer(iY) + (1+rM)*A0 - Astar - w2 - pE(jE) - ...
                            pFactor * Estar * Wstar * Pstar^(1+phiP);
        else
            r = rM + ((A0/ASmooth)/(1+(A0/ASmooth)^2)+0.5)*(rP-rM);
            c = Y0*(1-Deltay*D0) + YTransfer(iY) + (1+r)*A0 - Astar - w2 - pE(jE) - ...
                            pFactor * Estar * Wstar * Pstar^(1+phiP);
        end

        if c <= lowC || Wstar <= lowW
            v1Fine = -1e9;
        else        
            V0C1 = 0;
            V0C2 = 0;
            for jY = 1:nY
                V0C1 = V0C1 + TransY(iY,jY) * ( (1-tA)*((1-tW)*v0(jE, jA  ,jW  ,jY,1)   ...
                                                        +  tW *v0(jE, jA  ,jW+1,jY,1))  ...
                                                 + tA *((1-tW)*v0(jE, jA+1,jW  ,jY,1)   ...
                                                        +  tW *v0(jE, jA+1,jW+1,jY,1)));
                V0C2 = V0C2 + TransY(iY,jY) * ( (1-tA)*((1-tP)*v0(jE, jA  ,jP  ,jY,2)   ...
                                                        +  tP *v0(jE, jA  ,jP+1,jY,2))  ...
                                                 + tA *((1-tP)*v0(jE, jA+1,jP  ,jY,2)   ...
                                                        +  tP *v0(jE, jA+1,jP+1,jY,2)));
            end
            v1Fine = (((c-lowC)/scalefactor)^(1-gamma) - 1)/(1-gamma) + UWPart2 + rho2 * (a*Estar*V0C2 + (1-a*Estar)*V0C1);
        end

    end    
end
   

function [v1Fine,Estar,Astar,Wstar,Pstar,Istar,...
                       AstarMov,WstarMov,PstarMov,...
                       AstarStay,WstarStay,PstarStay] = MonotoniceV1_W(iE,iA,iW,iY,iD)
    
    Estar = PolicyFineE(iE,iA,iW,iY,iD);
    Astar = PolicyFineA(iE,iA,iW,iY,iD);
    Wstar = PolicyFineW(iE,iA,iW,iY,iD);
    Pstar = PolicyFineP(iE,iA,iW,iY,iD);
    Istar = PolicyFineI(iE,iA,iW,iY,iD);
    v1Fine = v1(iE,iA,iW,iY,iD);
    AstarMov = PolAMovHelp(iE,iA,iW,iY,iD);
    WstarMov = PolWMovHelp(iE,iA,iW,iY,iD);
    PstarMov = PolPMovHelp(iE,iA,iW,iY,iD);
    AstarStay = PolicyStayFineA(iE,iA,iW,iY,iD);
    WstarStay = PolicyStayFineW(iE,iA,iW,iY,iD);
    PstarStay = PolicyStayFineP(iE,iA,iW,iY,iD);   
    
    
    if MonotonicityW(iE,iA,iW,iY,iD) == 0
        A0 = Agrid(iA);
        W0 = Wgrid(iW);
        Y0 = Ygrid(iY);
        D0 = Dgrid(iD);
        
        Estar = PolicyFineE(iE,iA,iW-1,iY,iD);
        Astar = PolicyFineA(iE,iA,iW-1,iY,iD);
        Wstar = PolicyFineW(iE,iA,iW-1,iY,iD);
        Pstar = PolicyFineP(iE,iA,iW-1,iY,iD);
        Istar = PolicyFineI(iE,iA,iW-1,iY,iD);
        if Istar == 1
            AstarMov = Astar;
            WstarMov = Wstar;
            PstarMov = Pstar;
        else
            AstarStay = Astar;
            WstarStay = Wstar;
            PstarStay = Pstar;
        end
    
        jE = 0;
        for kE = 2:nE
            if Egrid(kE) > Estar
                jE = kE - 1;
                break
            elseif kE == nE
                jE = nE-1;
            end
        end
        
        jP = 0;
        for kW = 2:nW
            if Wgrid(kW) > Pstar*Wstar
                jP = kW - 1;
                break
            elseif kW == nW
                jP = nW-1;
            end
        end

        jA = 0;
        for kA = 2:nA
            if Agrid(kA) > Astar
                jA = kA - 1;
                break
            elseif kA == nA
                jA = nA-1;
            end
        end
    
        jW = 0;
        for kW = 2:nW
            if Wgrid(kW) > Wstar
                jW = kW - 1;
                break
            elseif kW == nW
                jW = nW-1;
            end
        end

        tA = (Astar - Agrid(jA))/(Agrid(jA+1)-Agrid(jA));
        tW = (Wstar - Wgrid(jW))/(Wgrid(jW+1)-Wgrid(jW));
        tP = (Pstar*Wstar - Wgrid(jP))/(Wgrid(jP+1)-Wgrid(jP));
       
        w = Wstar - (1-delta)*(1-DeltaW*Istar)*W0;
        if w <= -wSmooth
            winvest = kappaZ;
        elseif w > wSmooth
            winvest = 1;
        else
            winvest = kappaZ + ((w/wSmooth)/(1+(w/wSmooth)^2)+0.5)*(1-kappaZ);
        end
        w2 = w*winvest;
    
        if Wstar > lowW
            UWPart2 = theta * ((( Wstar-lowW )/scalefactor)^(1-beta)-1)/(1-beta);
        else
            UWPart2 = -1e6;
        end
    
        if A0 > ASmooth
            c = Y0*(1-Deltay*D0) + YTransfer(iY) + (1+rP)*A0 - Astar - w2 - pE(jE) - ...
                            pFactor * Estar * Wstar * Pstar^(1+phiP);
        elseif A0 <= -ASmooth
            c = Y0*(1-Deltay*D0) + YTransfer(iY) + (1+rM)*A0 - Astar - w2 - pE(jE) - ...
                            pFactor * Estar * Wstar * Pstar^(1+phiP);
        else
            r = rM + ((A0/ASmooth)/(1+(A0/ASmooth)^2)+0.5)*(rP-rM);
            c = Y0*(1-Deltay*D0) + YTransfer(iY) + (1+r)*A0 - Astar - w2 - pE(jE) - ...
                            pFactor * Estar * Wstar * Pstar^(1+phiP);
        end

        if c <= lowC || Wstar <= lowW
            v1Fine = -1e9;
        else        
            V0C1 = 0;
            V0C2 = 0;
            for jY = 1:nY
                V0C1 = V0C1 + TransY(iY,jY) * ( (1-tA)*((1-tW)*v0(jE, jA  ,jW  ,jY,1)   ...
                                                        +  tW *v0(jE, jA  ,jW+1,jY,1))  ...
                                                 + tA *((1-tW)*v0(jE, jA+1,jW  ,jY,1)   ...
                                                        +  tW *v0(jE, jA+1,jW+1,jY,1)));
                V0C2 = V0C2 + TransY(iY,jY) * ( (1-tA)*((1-tP)*v0(jE, jA  ,jP  ,jY,2)   ...
                                                        +  tP *v0(jE, jA  ,jP+1,jY,2))  ...
                                                 + tA *((1-tP)*v0(jE, jA+1,jP  ,jY,2)   ...
                                                        +  tP *v0(jE, jA+1,jP+1,jY,2)));
            end
            v1Fine = (((c-lowC)/scalefactor)^(1-gamma) - 1)/(1-gamma) + UWPart2 + rho2 * (a*Estar*V0C2 + (1-a*Estar)*V0C1);
        end

    end    
end
  

function [v2,Astar,Wstar,Pstar] = MonotoniceWMov(iE,iA,iW,iY,iD)
    if iW == 1 || v1MovHelp(iE,iA,iW,iY,iD) > v1MovHelp(iE,iA,max(iW-1,1),iY,iD)
        Astar = PolAMovHelp(iE,iA,iW,iY,iD);
        Wstar = PolWMovHelp(iE,iA,iW,iY,iD);
        Pstar = PolPMovHelp(iE,iA,iW,iY,iD);
        v2 = v1MovHelp(iE,iA,iW,iY,iD);
    else
        A0 = Agrid(iA);
        W0 = Wgrid(iW);
        Y0 = Ygrid(iY);
        D0 = Dgrid(iD);
        
        Astar = PolAMovHelp(iE,iA,iW-1,iY,iD);
        Wstar = PolWMovHelp(iE,iA,iW-1,iY,iD);
        Pstar = PolPMovHelp(iE,iA,iW-1,iY,iD);
        Estar = Egrid(iE);
        
        jP = 0;
        for kW = 2:nW
            if Wgrid(kW) > Pstar*Wstar
                jP = kW - 1;
                break
            elseif kW == nW
                jP = nW-1;
            end
        end

        jA = 0;
        for kA = 2:nA
            if Agrid(kA) > Astar
                jA = kA - 1;
                break
            elseif kA == nA
                jA = nA-1;
            end
        end
    
        jW = 0;
        for kW = 2:nW
            if Wgrid(kW) > Wstar
                jW = kW - 1;
                break
            elseif kW == nW
                jW = nW-1;
            end
        end

        tA = (Astar - Agrid(jA))/(Agrid(jA+1)-Agrid(jA));
        tW = (Wstar - Wgrid(jW))/(Wgrid(jW+1)-Wgrid(jW));
        tP = (Pstar*Wstar - Wgrid(jP))/(Wgrid(jP+1)-Wgrid(jP));
       
        w = Wstar - (1-delta)*(1-DeltaW)*W0;
        if w <= -wSmooth
            winvest = kappaZ;
        elseif w > wSmooth
            winvest = 1;
        else
            winvest = kappaZ + ((w/wSmooth)/(1+(w/wSmooth)^2)+0.5)*(1-kappaZ);
        end
        w2 = w*winvest;
    
        if Wstar > lowW
            UWPart2 = theta * ((( Wstar-lowW )/scalefactor)^(1-beta)-1)/(1-beta);
        else
            UWPart2 = -1e6;
        end
    
        if A0 > ASmooth
            c = Y0*(1-Deltay*D0) + YTransfer(iY) + (1+rP)*A0 - Astar - w2 - pE(iE) - ...
                            pFactor * Estar * Wstar * Pstar^(1+phiP);
        elseif A0 <= -ASmooth
            c = Y0*(1-Deltay*D0) + YTransfer(iY) + (1+rM)*A0 - Astar - w2 - pE(iE) - ...
                            pFactor * Estar * Wstar * Pstar^(1+phiP);
        else
            r = rM + ((A0/ASmooth)/(1+(A0/ASmooth)^2)+0.5)*(rP-rM);
            c = Y0*(1-Deltay*D0) + YTransfer(iY) + (1+r)*A0 - Astar - w2 - pE(iE) - ...
                            pFactor * Estar * Wstar * Pstar^(1+phiP);
        end

        if c <= lowC || Wstar <= lowW
            v2 = -1e9;
        else        
            V0C1 = 0;
            V0C2 = 0;
            for jY = 1:nY
                V0C1 = V0C1 + TransY(iY,jY) * ( (1-tA)*((1-tW)*v0(iE, jA  ,jW  ,jY,1)   ...
                                                        +  tW *v0(iE, jA  ,jW+1,jY,1))  ...
                                                 + tA *((1-tW)*v0(iE, jA+1,jW  ,jY,1)   ...
                                                        +  tW *v0(iE, jA+1,jW+1,jY,1)));
                V0C2 = V0C2 + TransY(iY,jY) * ( (1-tA)*((1-tP)*v0(iE, jA  ,jP  ,jY,2)   ...
                                                        +  tP *v0(iE, jA  ,jP+1,jY,2))  ...
                                                 + tA *((1-tP)*v0(iE, jA+1,jP  ,jY,2)   ...
                                                        +  tP *v0(iE, jA+1,jP+1,jY,2)));
            end
            v2 = (((c-lowC)/scalefactor)^(1-gamma) - 1)/(1-gamma) + UWPart2 + rho2 * (a*Estar*V0C2 + (1-a*Estar)*V0C1);
        end

    end    
end
  
function [v2,Astar,Wstar,Pstar] = MonotoniceAMov(iE,iA,iW,iY,iD)
    v2=0;
    Astar=0;
    Wstar=0;
    Pstar=0;
    
    if iA == 1 || v1MovHelp(iE,iA,iW,iY,iD) > v1MovHelp(iE,max(iA-1,1),iW,iY,iD)
        Astar = PolAMovHelp(iE,iA,iW,iY,iD);
        Wstar = PolWMovHelp(iE,iA,iW,iY,iD);
        Pstar = PolPMovHelp(iE,iA,iW,iY,iD);
        v2 = v1MovHelp(iE,iA,iW,iY,iD);
    else
        A0 = Agrid(iA);
        W0 = Wgrid(iW);
        Y0 = Ygrid(iY);
        D0 = Dgrid(iD);
        
        Astar = PolAMovHelp(iE,iA-1,iW,iY,iD);
        Wstar = PolWMovHelp(iE,iA-1,iW,iY,iD);
        Pstar = PolPMovHelp(iE,iA-1,iW,iY,iD);
        Estar = Egrid(iE); 
        jP = 0;
        for kW = 2:nW
            if Wgrid(kW) > Pstar*Wstar
                jP = kW - 1;
                break
            elseif kW == nW
                jP = nW-1;
            end
        end

        jA = 0;
        for kA = 2:nA
            if Agrid(kA) > Astar
                jA = kA - 1;
                break
            elseif kA == nA
                jA = nA-1;
            end
        end
    
        jW = 0;
        for kW = 2:nW
            if Wgrid(kW) > Wstar
                jW = kW - 1;
                break
            elseif kW == nW
                jW = nW-1;
            end
        end

        tA = (Astar - Agrid(jA))/(Agrid(jA+1)-Agrid(jA));
        tW = (Wstar - Wgrid(jW))/(Wgrid(jW+1)-Wgrid(jW));
        tP = (Pstar*Wstar - Wgrid(jP))/(Wgrid(jP+1)-Wgrid(jP));
       
        w = Wstar - (1-delta)*(1-DeltaW)*W0;
        if w <= -wSmooth
            winvest = kappaZ;
        elseif w > wSmooth
            winvest = 1;
        else
            winvest = kappaZ + ((w/wSmooth)/(1+(w/wSmooth)^2)+0.5)*(1-kappaZ);
        end
        w2 = w*winvest;
    
        if Wstar > lowW
            UWPart2 = theta * ((( Wstar-lowW )/scalefactor)^(1-beta)-1)/(1-beta);
        else
            UWPart2 = -1e6;
        end
    
        if A0 > ASmooth
            c = Y0*(1-Deltay*D0) + YTransfer(iY) + (1+rP)*A0 - Astar - w2 - pE(iE) - ...
                            pFactor * Estar * Wstar * Pstar^(1+phiP);
        elseif A0 <= -ASmooth
            c = Y0*(1-Deltay*D0) + YTransfer(iY) + (1+rM)*A0 - Astar - w2 - pE(iE) - ...
                            pFactor * Estar * Wstar * Pstar^(1+phiP);
        else
            r = rM + ((A0/ASmooth)/(1+(A0/ASmooth)^2)+0.5)*(rP-rM);
            c = Y0*(1-Deltay*D0) + YTransfer(iY) + (1+r)*A0 - Astar - w2 - pE(iE) - ...
                            pFactor * Estar * Wstar * Pstar^(1+phiP);
        end

        if c <= lowC || Wstar <= lowW
            v2 = -1e9;
        else        
            V0C1 = 0;
            V0C2 = 0;
            for jY = 1:nY
                V0C1 = V0C1 + TransY(iY,jY) * ( (1-tA)*((1-tW)*v0(iE, jA  ,jW  ,jY,1)   ...
                                                        +  tW *v0(iE, jA  ,jW+1,jY,1))  ...
                                                 + tA *((1-tW)*v0(iE, jA+1,jW  ,jY,1)   ...
                                                        +  tW *v0(iE, jA+1,jW+1,jY,1)));
                V0C2 = V0C2 + TransY(iY,jY) * ( (1-tA)*((1-tP)*v0(iE, jA  ,jP  ,jY,2)   ...
                                                        +  tP *v0(iE, jA  ,jP+1,jY,2))  ...
                                                 + tA *((1-tP)*v0(iE, jA+1,jP  ,jY,2)   ...
                                                        +  tP *v0(iE, jA+1,jP+1,jY,2)));
            end
            v2 = (((c-lowC)/scalefactor)^(1-gamma) - 1)/(1-gamma) + UWPart2 + rho2 * (a*Estar*V0C2 + (1-a*Estar)*V0C1);
        end

    end    
end

function [v2,Astar,Wstar,Pstar] = MonotoniceDMov(iE,iA,iW,iY,iD)
    v2=0;
    Astar=0;
    Wstar=0;
    Pstar=0;
    
    if iD == 2 || v1MovHelp(iE,iA,iW,iY,1) > v1MovHelp(iE,iA,iW,iY,2)
        Astar = PolAMovHelp(iE,iA,iW,iY,iD);
        Wstar = PolWMovHelp(iE,iA,iW,iY,iD);
        Pstar = PolPMovHelp(iE,iA,iW,iY,iD);
        v2 = v1MovHelp(iE,iA,iW,iY,iD);
    else
        A0 = Agrid(iA);
        W0 = Wgrid(iW);
        Y0 = Ygrid(iY);
        D0 = Dgrid(iD);
        
        Astar = PolAMovHelp(iE,iA,iW,iY,2);
        Wstar = PolWMovHelp(iE,iA,iW,iY,2);
        Pstar = PolPMovHelp(iE,iA,iW,iY,2);
        Estar = Egrid(iE); 
        jP = 0;
        for kW = 2:nW
            if Wgrid(kW) > Pstar*Wstar
                jP = kW - 1;
                break
            elseif kW == nW
                jP = nW-1;
            end
        end

        jA = 0;
        for kA = 2:nA
            if Agrid(kA) > Astar
                jA = kA - 1;
                break
            elseif kA == nA
                jA = nA-1;
            end
        end
    
        jW = 0;
        for kW = 2:nW
            if Wgrid(kW) > Wstar
                jW = kW - 1;
                break
            elseif kW == nW
                jW = nW-1;
            end
        end

        tA = (Astar - Agrid(jA))/(Agrid(jA+1)-Agrid(jA));
        tW = (Wstar - Wgrid(jW))/(Wgrid(jW+1)-Wgrid(jW));
        tP = (Pstar*Wstar - Wgrid(jP))/(Wgrid(jP+1)-Wgrid(jP));
       
        w = Wstar - (1-delta)*(1-DeltaW)*W0;
        if w <= -wSmooth
            winvest = kappaZ;
        elseif w > wSmooth
            winvest = 1;
        else
            winvest = kappaZ + ((w/wSmooth)/(1+(w/wSmooth)^2)+0.5)*(1-kappaZ);
        end
        w2 = w*winvest;
    
        if Wstar > lowW
            UWPart2 = theta * ((( Wstar-lowW )/scalefactor)^(1-beta)-1)/(1-beta);
        else
            UWPart2 = -1e6;
        end
    
        if A0 > ASmooth
            c = Y0*(1-Deltay*D0) + YTransfer(iY) + (1+rP)*A0 - Astar - w2 - pE(iE) - ...
                            pFactor * Estar * Wstar * Pstar^(1+phiP);
        elseif A0 <= -ASmooth
            c = Y0*(1-Deltay*D0) + YTransfer(iY) + (1+rM)*A0 - Astar - w2 - pE(iE) - ...
                            pFactor * Estar * Wstar * Pstar^(1+phiP);
        else
            r = rM + ((A0/ASmooth)/(1+(A0/ASmooth)^2)+0.5)*(rP-rM);
            c = Y0*(1-Deltay*D0) + YTransfer(iY) + (1+r)*A0 - Astar - w2 - pE(iE) - ...
                            pFactor * Estar * Wstar * Pstar^(1+phiP);
        end

        if c <= lowC || Wstar <= lowW
            v2 = -1e9;
        else        
            V0C1 = 0;
            V0C2 = 0;
            for jY = 1:nY
                V0C1 = V0C1 + TransY(iY,jY) * ( (1-tA)*((1-tW)*v0(iE, jA  ,jW  ,jY,1)   ...
                                                        +  tW *v0(iE, jA  ,jW+1,jY,1))  ...
                                                 + tA *((1-tW)*v0(iE, jA+1,jW  ,jY,1)   ...
                                                        +  tW *v0(iE, jA+1,jW+1,jY,1)));
                V0C2 = V0C2 + TransY(iY,jY) * ( (1-tA)*((1-tP)*v0(iE, jA  ,jP  ,jY,2)   ...
                                                        +  tP *v0(iE, jA  ,jP+1,jY,2))  ...
                                                 + tA *((1-tP)*v0(iE, jA+1,jP  ,jY,2)   ...
                                                        +  tP *v0(iE, jA+1,jP+1,jY,2)));
            end
            v2 = (((c-lowC)/scalefactor)^(1-gamma) - 1)/(1-gamma) + UWPart2 + rho2 * (a*Estar*V0C2 + (1-a*Estar)*V0C1);
        end

    end    
end

function [v2,Astar,Wstar,Pstar] = MonotoniceYMov(iE,iA,iW,iY,iD)
    v2=0;
    Astar=0;
    Wstar=0;
    Pstar=0;
    
    if iY == 1 || v1MovHelp(iE,iA,iW,iY,iD) > v1MovHelp(iE,iA,iW,max(iY-1,1),iD)
        Astar = PolAMovHelp(iE,iA,iW,iY,iD);
        Wstar = PolWMovHelp(iE,iA,iW,iY,iD);
        Pstar = PolPMovHelp(iE,iA,iW,iY,iD);
        v2 = v1MovHelp(iE,iA,iW,iY,iD);
    else
        A0 = Agrid(iA);
        W0 = Wgrid(iW);
        Y0 = Ygrid(iY);
        D0 = Dgrid(iD);
        
        Astar = PolAMovHelp(iE,iA,iW,iY-1,iD);
        Wstar = PolWMovHelp(iE,iA,iW,iY-1,iD);
        Pstar = PolPMovHelp(iE,iA,iW,iY-1,iD);
        Estar = Egrid(iE); 
        jP = 0;
        for kW = 2:nW
            if Wgrid(kW) > Pstar*Wstar
                jP = kW - 1;
                break
            elseif kW == nW
                jP = nW-1;
            end
        end

        jA = 0;
        for kA = 2:nA
            if Agrid(kA) > Astar
                jA = kA - 1;
                break
            elseif kA == nA
                jA = nA-1;
            end
        end
    
        jW = 0;
        for kW = 2:nW
            if Wgrid(kW) > Wstar
                jW = kW - 1;
                break
            elseif kW == nW
                jW = nW-1;
            end
        end

        tA = (Astar - Agrid(jA))/(Agrid(jA+1)-Agrid(jA));
        tW = (Wstar - Wgrid(jW))/(Wgrid(jW+1)-Wgrid(jW));
        tP = (Pstar*Wstar - Wgrid(jP))/(Wgrid(jP+1)-Wgrid(jP));
       
        w = Wstar - (1-delta)*(1-DeltaW)*W0;
        if w <= -wSmooth
            winvest = kappaZ;
        elseif w > wSmooth
            winvest = 1;
        else
            winvest = kappaZ + ((w/wSmooth)/(1+(w/wSmooth)^2)+0.5)*(1-kappaZ);
        end
        w2 = w*winvest;
    
        if Wstar > lowW
            UWPart2 = theta * ((( Wstar-lowW )/scalefactor)^(1-beta)-1)/(1-beta);
        else
            UWPart2 = -1e6;
        end
    
        if A0 > ASmooth
            c = Y0*(1-Deltay*D0) + YTransfer(iY) + (1+rP)*A0 - Astar - w2 - pE(iE) - ...
                            pFactor * Estar * Wstar * Pstar^(1+phiP);
        elseif A0 <= -ASmooth
            c = Y0*(1-Deltay*D0) + YTransfer(iY) + (1+rM)*A0 - Astar - w2 - pE(iE) - ...
                            pFactor * Estar * Wstar * Pstar^(1+phiP);
        else
            r = rM + ((A0/ASmooth)/(1+(A0/ASmooth)^2)+0.5)*(rP-rM);
            c = Y0*(1-Deltay*D0) + YTransfer(iY) + (1+r)*A0 - Astar - w2 - pE(iE) - ...
                            pFactor * Estar * Wstar * Pstar^(1+phiP);
        end

        if c <= lowC || Wstar <= lowW
            v2 = -1e9;
        else        
            V0C1 = 0;
            V0C2 = 0;
            for jY = 1:nY
                V0C1 = V0C1 + TransY(iY,jY) * ( (1-tA)*((1-tW)*v0(iE, jA  ,jW  ,jY,1)   ...
                                                        +  tW *v0(iE, jA  ,jW+1,jY,1))  ...
                                                 + tA *((1-tW)*v0(iE, jA+1,jW  ,jY,1)   ...
                                                        +  tW *v0(iE, jA+1,jW+1,jY,1)));
                V0C2 = V0C2 + TransY(iY,jY) * ( (1-tA)*((1-tP)*v0(iE, jA  ,jP  ,jY,2)   ...
                                                        +  tP *v0(iE, jA  ,jP+1,jY,2))  ...
                                                 + tA *((1-tP)*v0(iE, jA+1,jP  ,jY,2)   ...
                                                        +  tP *v0(iE, jA+1,jP+1,jY,2)));
            end
            v2 = (((c-lowC)/scalefactor)^(1-gamma) - 1)/(1-gamma) + UWPart2 + rho2 * (a*Estar*V0C2 + (1-a*Estar)*V0C1);
        end

    end    
end


function [v2,Astar,Wstar,Pstar] = MonotoniceWStay(iE,iA,iW,iY,iD)
    if iW == 1 || v1StayFine(iE,iA,iW,iY,iD) > v1StayFine(iE,iA,max(iW-1,1),iY,iD)
        Astar = PolicyStayFineA(iE,iA,iW,iY,iD);
        Wstar = PolicyStayFineW(iE,iA,iW,iY,iD);
        Pstar = PolicyStayFineP(iE,iA,iW,iY,iD);
        v2 = v1StayFine(iE,iA,iW,iY,iD);
    else
        A0 = Agrid(iA);
        W0 = Wgrid(iW);
        Y0 = Ygrid(iY);
        D0 = Dgrid(iD);
        
        Astar = PolicyStayFineA(iE,iA,iW-1,iY,iD);
        Wstar = PolicyStayFineW(iE,iA,iW-1,iY,iD);
        Pstar = PolicyStayFineP(iE,iA,iW-1,iY,iD);
        Estar = Egrid(iE);
        
        jP = 0;
        for kW = 2:nW
            if Wgrid(kW) > Pstar*Wstar
                jP = kW - 1;
                break
            elseif kW == nW
                jP = nW-1;
            end
        end

        jA = 0;
        for kA = 2:nA
            if Agrid(kA) > Astar
                jA = kA - 1;
                break
            elseif kA == nA
                jA = nA-1;
            end
        end
    
        jW = 0;
        for kW = 2:nW
            if Wgrid(kW) > Wstar
                jW = kW - 1;
                break
            elseif kW == nW
                jW = nW-1;
            end
        end

        tA = (Astar - Agrid(jA))/(Agrid(jA+1)-Agrid(jA));
        tW = (Wstar - Wgrid(jW))/(Wgrid(jW+1)-Wgrid(jW));
        tP = (Pstar*Wstar - Wgrid(jP))/(Wgrid(jP+1)-Wgrid(jP));
       
        w = Wstar - (1-delta)*W0;
        if w <= -wSmooth
            winvest = kappaZ;
        elseif w > wSmooth
            winvest = 1;
        else
            winvest = kappaZ + ((w/wSmooth)/(1+(w/wSmooth)^2)+0.5)*(1-kappaZ);
        end
        w2 = w*winvest;
    
        if Wstar > lowW
            UWPart2 = theta * ((( Wstar-lowW )/scalefactor)^(1-beta)-1)/(1-beta);
        else
            UWPart2 = -1e6;
        end
    
        if A0 > ASmooth
            c = Y0*(1-Deltay*D0) + YTransfer(iY) + (1+rP)*A0 - Astar - w2 - pE(iE) - ...
                            pFactor * Estar * Wstar * Pstar^(1+phiP);
        elseif A0 <= -ASmooth
            c = Y0*(1-Deltay*D0) + YTransfer(iY) + (1+rM)*A0 - Astar - w2 - pE(iE) - ...
                            pFactor * Estar * Wstar * Pstar^(1+phiP);
        else
            r = rM + ((A0/ASmooth)/(1+(A0/ASmooth)^2)+0.5)*(rP-rM);
            c = Y0*(1-Deltay*D0) + YTransfer(iY) + (1+r)*A0 - Astar - w2 - pE(iE) - ...
                            pFactor * Estar * Wstar * Pstar^(1+phiP);
        end

        if c <= lowC || Wstar <= lowW
            v2 = -1e9;
        else        
            V0C1 = 0;
            V0C2 = 0;
            for jY = 1:nY
                V0C1 = V0C1 + TransY(iY,jY) * ( (1-tA)*((1-tW)*v0(iE, jA  ,jW  ,jY,1)   ...
                                                        +  tW *v0(iE, jA  ,jW+1,jY,1))  ...
                                                 + tA *((1-tW)*v0(iE, jA+1,jW  ,jY,1)   ...
                                                        +  tW *v0(iE, jA+1,jW+1,jY,1)));
                V0C2 = V0C2 + TransY(iY,jY) * ( (1-tA)*((1-tP)*v0(iE, jA  ,jP  ,jY,2)   ...
                                                        +  tP *v0(iE, jA  ,jP+1,jY,2))  ...
                                                 + tA *((1-tP)*v0(iE, jA+1,jP  ,jY,2)   ...
                                                        +  tP *v0(iE, jA+1,jP+1,jY,2)));
            end
            v2 = (((c-lowC)/scalefactor)^(1-gamma) - 1)/(1-gamma) + UWPart2 + rho2 * (a*Estar*V0C2 + (1-a*Estar)*V0C1);
        end
    end    
end
  
function [v2,Astar,Wstar,Pstar] = MonotoniceAStay(iE,iA,iW,iY,iD)
    if iA == 1 || v1StayFine(iE,iA,iW,iY,iD) > v1StayFine(iE,max(iA-1,1),iW,iY,iD)
        Astar = PolicyStayFineA(iE,iA,iW,iY,iD);
        Wstar = PolicyStayFineW(iE,iA,iW,iY,iD);
        Pstar = PolicyStayFineP(iE,iA,iW,iY,iD);
        v2 = v1StayFine(iE,iA,iW,iY,iD);
    else
        A0 = Agrid(iA);
        W0 = Wgrid(iW);
        Y0 = Ygrid(iY);
        D0 = Dgrid(iD);
        
        Astar = PolicyStayFineA(iE,iA-1,iW,iY,iD);
        Wstar = PolicyStayFineW(iE,iA-1,iW,iY,iD);
        Pstar = PolicyStayFineP(iE,iA-1,iW,iY,iD);
        Estar = Egrid(iE);
        
        jP = 0;
        for kW = 2:nW
            if Wgrid(kW) > Pstar*Wstar
                jP = kW - 1;
                break
            elseif kW == nW
                jP = nW-1;
            end
        end

        jA = 0;
        for kA = 2:nA
            if Agrid(kA) > Astar
                jA = kA - 1;
                break
            elseif kA == nA
                jA = nA-1;
            end
        end
    
        jW = 0;
        for kW = 2:nW
            if Wgrid(kW) > Wstar
                jW = kW - 1;
                break
            elseif kW == nW
                jW = nW-1;
            end
        end

        tA = (Astar - Agrid(jA))/(Agrid(jA+1)-Agrid(jA));
        tW = (Wstar - Wgrid(jW))/(Wgrid(jW+1)-Wgrid(jW));
        tP = (Pstar*Wstar - Wgrid(jP))/(Wgrid(jP+1)-Wgrid(jP));
       
        w = Wstar - (1-delta)*W0;
        if w <= -wSmooth
            winvest = kappaZ;
        elseif w > wSmooth
            winvest = 1;
        else
            winvest = kappaZ + ((w/wSmooth)/(1+(w/wSmooth)^2)+0.5)*(1-kappaZ);
        end
        w2 = w*winvest;
    
        if Wstar > lowW
            UWPart2 = theta * ((( Wstar-lowW )/scalefactor)^(1-beta)-1)/(1-beta);
        else
            UWPart2 = -1e6;
        end
    
        if A0 > ASmooth
            c = Y0*(1-Deltay*D0) + YTransfer(iY) + (1+rP)*A0 - Astar - w2 - pE(iE) - ...
                            pFactor * Estar * Wstar * Pstar^(1+phiP);
        elseif A0 <= -ASmooth
            c = Y0*(1-Deltay*D0) + YTransfer(iY) + (1+rM)*A0 - Astar - w2 - pE(iE) - ...
                            pFactor * Estar * Wstar * Pstar^(1+phiP);
        else
            r = rM + ((A0/ASmooth)/(1+(A0/ASmooth)^2)+0.5)*(rP-rM);
            c = Y0*(1-Deltay*D0) + YTransfer(iY) + (1+r)*A0 - Astar - w2 - pE(iE) - ...
                            pFactor * Estar * Wstar * Pstar^(1+phiP);
        end

        if c <= lowC || Wstar <= lowW
            v2 = -1e9;
        else        
            V0C1 = 0;
            V0C2 = 0;
            for jY = 1:nY
                V0C1 = V0C1 + TransY(iY,jY) * ( (1-tA)*((1-tW)*v0(iE, jA  ,jW  ,jY,1)   ...
                                                        +  tW *v0(iE, jA  ,jW+1,jY,1))  ...
                                                 + tA *((1-tW)*v0(iE, jA+1,jW  ,jY,1)   ...
                                                        +  tW *v0(iE, jA+1,jW+1,jY,1)));
                V0C2 = V0C2 + TransY(iY,jY) * ( (1-tA)*((1-tP)*v0(iE, jA  ,jP  ,jY,2)   ...
                                                        +  tP *v0(iE, jA  ,jP+1,jY,2))  ...
                                                 + tA *((1-tP)*v0(iE, jA+1,jP  ,jY,2)   ...
                                                        +  tP *v0(iE, jA+1,jP+1,jY,2)));
            end
            v2 = (((c-lowC)/scalefactor)^(1-gamma) - 1)/(1-gamma) + UWPart2 + rho2 * (a*Estar*V0C2 + (1-a*Estar)*V0C1);
        end
    end    
end
 
function [v2,Astar,Wstar,Pstar] = MonotoniceDStay(iE,iA,iW,iY,iD)
    if iD == 2 || v1StayFine(iE,iA,iW,iY,1) > v1StayFine(iE,iA,iW,iY,2)
        Astar = PolicyStayFineA(iE,iA,iW,iY,iD);
        Wstar = PolicyStayFineW(iE,iA,iW,iY,iD);
        Pstar = PolicyStayFineP(iE,iA,iW,iY,iD);
        v2 = v1StayFine(iE,iA,iW,iY,iD);
    else
        A0 = Agrid(iA);
        W0 = Wgrid(iW);
        Y0 = Ygrid(iY);
        D0 = Dgrid(iD);
        
        Astar = PolicyStayFineA(iE,iA,iW,iY,2);
        Wstar = PolicyStayFineW(iE,iA,iW,iY,2);
        Pstar = PolicyStayFineP(iE,iA,iW,iY,2);
        Estar = Egrid(iE);
        
        jP = 0;
        for kW = 2:nW
            if Wgrid(kW) > Pstar*Wstar
                jP = kW - 1;
                break
            elseif kW == nW
                jP = nW-1;
            end
        end

        jA = 0;
        for kA = 2:nA
            if Agrid(kA) > Astar
                jA = kA - 1;
                break
            elseif kA == nA
                jA = nA-1;
            end
        end
    
        jW = 0;
        for kW = 2:nW
            if Wgrid(kW) > Wstar
                jW = kW - 1;
                break
            elseif kW == nW
                jW = nW-1;
            end
        end

        tA = (Astar - Agrid(jA))/(Agrid(jA+1)-Agrid(jA));
        tW = (Wstar - Wgrid(jW))/(Wgrid(jW+1)-Wgrid(jW));
        tP = (Pstar*Wstar - Wgrid(jP))/(Wgrid(jP+1)-Wgrid(jP));
       
        w = Wstar - (1-delta)*W0;
        if w <= -wSmooth
            winvest = kappaZ;
        elseif w > wSmooth
            winvest = 1;
        else
            winvest = kappaZ + ((w/wSmooth)/(1+(w/wSmooth)^2)+0.5)*(1-kappaZ);
        end
        w2 = w*winvest;
    
        if Wstar > lowW
            UWPart2 = theta * ((( Wstar-lowW )/scalefactor)^(1-beta)-1)/(1-beta);
        else
            UWPart2 = -1e6;
        end
    
        if A0 > ASmooth
            c = Y0*(1-Deltay*D0) + YTransfer(iY) + (1+rP)*A0 - Astar - w2 - pE(iE) - ...
                            pFactor * Estar * Wstar * Pstar^(1+phiP);
        elseif A0 <= -ASmooth
            c = Y0*(1-Deltay*D0) + YTransfer(iY) + (1+rM)*A0 - Astar - w2 - pE(iE) - ...
                            pFactor * Estar * Wstar * Pstar^(1+phiP);
        else
            r = rM + ((A0/ASmooth)/(1+(A0/ASmooth)^2)+0.5)*(rP-rM);
            c = Y0*(1-Deltay*D0) + YTransfer(iY) + (1+r)*A0 - Astar - w2 - pE(iE) - ...
                            pFactor * Estar * Wstar * Pstar^(1+phiP);
        end

        if c <= lowC || Wstar <= lowW
            v2 = -1e9;
        else        
            V0C1 = 0;
            V0C2 = 0;
            for jY = 1:nY
                V0C1 = V0C1 + TransY(iY,jY) * ( (1-tA)*((1-tW)*v0(iE, jA  ,jW  ,jY,1)   ...
                                                        +  tW *v0(iE, jA  ,jW+1,jY,1))  ...
                                                 + tA *((1-tW)*v0(iE, jA+1,jW  ,jY,1)   ...
                                                        +  tW *v0(iE, jA+1,jW+1,jY,1)));
                V0C2 = V0C2 + TransY(iY,jY) * ( (1-tA)*((1-tP)*v0(iE, jA  ,jP  ,jY,2)   ...
                                                        +  tP *v0(iE, jA  ,jP+1,jY,2))  ...
                                                 + tA *((1-tP)*v0(iE, jA+1,jP  ,jY,2)   ...
                                                        +  tP *v0(iE, jA+1,jP+1,jY,2)));
            end
            v2 = (((c-lowC)/scalefactor)^(1-gamma) - 1)/(1-gamma) + UWPart2 + rho2 * (a*Estar*V0C2 + (1-a*Estar)*V0C1);
        end
    end    
end
 
function [v2,Astar,Wstar,Pstar] = MonotoniceYStay(iE,iA,iW,iY,iD)
    if iY == 1 || v1StayFine(iE,iA,iW,iY,iD) > v1StayFine(iE,iA,iW,max(iY-1,1),iD)
        Astar = PolicyStayFineA(iE,iA,iW,iY,iD);
        Wstar = PolicyStayFineW(iE,iA,iW,iY,iD);
        Pstar = PolicyStayFineP(iE,iA,iW,iY,iD);
        v2 = v1StayFine(iE,iA,iW,iY,iD);
    else
        A0 = Agrid(iA);
        W0 = Wgrid(iW);
        Y0 = Ygrid(iY);
        D0 = Dgrid(iD);
        
        Astar = PolicyStayFineA(iE,iA,iW,iY-1,iD);
        Wstar = PolicyStayFineW(iE,iA,iW,iY-1,iD);
        Pstar = PolicyStayFineP(iE,iA,iW,iY-1,iD);
        Estar = Egrid(iE);
        
        jP = 0;
        for kW = 2:nW
            if Wgrid(kW) > Pstar*Wstar
                jP = kW - 1;
                break
            elseif kW == nW
                jP = nW-1;
            end
        end

        jA = 0;
        for kA = 2:nA
            if Agrid(kA) > Astar
                jA = kA - 1;
                break
            elseif kA == nA
                jA = nA-1;
            end
        end
    
        jW = 0;
        for kW = 2:nW
            if Wgrid(kW) > Wstar
                jW = kW - 1;
                break
            elseif kW == nW
                jW = nW-1;
            end
        end

        tA = (Astar - Agrid(jA))/(Agrid(jA+1)-Agrid(jA));
        tW = (Wstar - Wgrid(jW))/(Wgrid(jW+1)-Wgrid(jW));
        tP = (Pstar*Wstar - Wgrid(jP))/(Wgrid(jP+1)-Wgrid(jP));
       
        w = Wstar - (1-delta)*W0;
        if w <= -wSmooth
            winvest = kappaZ;
        elseif w > wSmooth
            winvest = 1;
        else
            winvest = kappaZ + ((w/wSmooth)/(1+(w/wSmooth)^2)+0.5)*(1-kappaZ);
        end
        w2 = w*winvest;
    
        if Wstar > lowW
            UWPart2 = theta * ((( Wstar-lowW )/scalefactor)^(1-beta)-1)/(1-beta);
        else
            UWPart2 = -1e6;
        end
    
        if A0 > ASmooth
            c = Y0*(1-Deltay*D0) + YTransfer(iY) + (1+rP)*A0 - Astar - w2 - pE(iE) - ...
                            pFactor * Estar * Wstar * Pstar^(1+phiP);
        elseif A0 <= -ASmooth
            c = Y0*(1-Deltay*D0) + YTransfer(iY) + (1+rM)*A0 - Astar - w2 - pE(iE) - ...
                            pFactor * Estar * Wstar * Pstar^(1+phiP);
        else
            r = rM + ((A0/ASmooth)/(1+(A0/ASmooth)^2)+0.5)*(rP-rM);
            c = Y0*(1-Deltay*D0) + YTransfer(iY) + (1+r)*A0 - Astar - w2 - pE(iE) - ...
                            pFactor * Estar * Wstar * Pstar^(1+phiP);
        end

        if c <= lowC || Wstar <= lowW
            v2 = -1e9;
        else        
            V0C1 = 0;
            V0C2 = 0;
            for jY = 1:nY
                V0C1 = V0C1 + TransY(iY,jY) * ( (1-tA)*((1-tW)*v0(iE, jA  ,jW  ,jY,1)   ...
                                                        +  tW *v0(iE, jA  ,jW+1,jY,1))  ...
                                                 + tA *((1-tW)*v0(iE, jA+1,jW  ,jY,1)   ...
                                                        +  tW *v0(iE, jA+1,jW+1,jY,1)));
                V0C2 = V0C2 + TransY(iY,jY) * ( (1-tA)*((1-tP)*v0(iE, jA  ,jP  ,jY,2)   ...
                                                        +  tP *v0(iE, jA  ,jP+1,jY,2))  ...
                                                 + tA *((1-tP)*v0(iE, jA+1,jP  ,jY,2)   ...
                                                        +  tP *v0(iE, jA+1,jP+1,jY,2)));
            end
            v2 = (((c-lowC)/scalefactor)^(1-gamma) - 1)/(1-gamma) + UWPart2 + rho2 * (a*Estar*V0C2 + (1-a*Estar)*V0C1);
        end
    end    
end
 
%% ---------------------------------------------------------------------------------------------------------------------
%   Initialise Final result arrays
%-----------------------------------------------------------------------------------------------------------------------

Policy.E.Opt = zeros(nH0,nrho,na,nphiP,nE,nA,nW,nY,nD);
Policy.A.Opt = zeros(nH0,nrho,na,nphiP,nE,nA,nW,nY,nD);
Policy.W.Opt = zeros(nH0,nrho,na,nphiP,nE,nA,nW,nY,nD);
Policy.P.Opt = zeros(nH0,nrho,na,nphiP,nE,nA,nW,nY,nD);
Policy.I.Opt = int16(zeros(nH0,nrho,na,nphiP,nE,nA,nW,nY,nD));

Policy.c.Opt = zeros(nH0,nrho,na,nphiP,nE,nA,nW,nY,nD);
Policy.w.Opt = zeros(nH0,nrho,na,nphiP,nE,nA,nW,nY,nD);
Policy.Vul.Opt = zeros(nH0,nrho,na,nphiP,nE,nA,nW,nY,nD);
Policy.NetSavings.Opt = zeros(nH0,nrho,na,nphiP,nE,nA,nW,nY,nD);
Policy.NetWealthChange.Opt = zeros(nH0,nrho,na,nphiP,nE,nA,nW,nY,nD);

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
ValueFunction.Stay.Fine = zeros(nH0,nrho,na,nphiP,nE,nA,nW,nY,nD);
ValueFunction.Mov.Fine = zeros(nH0,nrho,na,nphiP,nA,nW,nY,nD);

%% ---------------------------------------------------------------------------------------------------------------------
%   Actual Algorithm
%-----------------------------------------------------------------------------------------------------------------------

for iH0 = 1:nH0
    for irho = 1:nrho
        for ia = 1:na
            for iphiP = 1:nphiP
                if Para.NothingChanged(iH0,irho,ia,iphiP) == 1
                    % Skip calculations and load results from Benchmark data
                else

                    fprintf("Start combination iH0 = %d | irho = %d | ia = %d | iphiP = %d \n",iH0,irho,ia,iphiP)

                    a = Grid.a(ia);
                    phiP = Grid.phiP(iphiP);
                    rho = Grid.rho(irho);
                    rho2 = 1/(1+rho); 

                    Agrid = squeeze(Grid.A(iH0,irho,ia,iphiP,:))';
                    Wgrid = squeeze(Grid.W(iH0,irho,ia,iphiP,:))';
                    Wgrid(2) = 0.5*(Wgrid(1)+Wgrid(3));
                    Grid.W(iH0,irho,ia,iphiP,:) = Wgrid;

                    Ygrid = Grid.Y(iH0,:);
                    TransY = squeeze(Grid.TransY(iH0,:,:));
                    YTransfer = squeeze(Grid.YTransfer_H0(iH0,:));

                    PolicyFineEOld = squeeze(PolicyDis.E.Dis(iH0,irho,ia,iphiP,:,:,:,:,:));
                    PolicyFineAOld = squeeze(PolicyDis.A.Dis(iH0,irho,ia,iphiP,:,:,:,:,:));
                    PolicyFineWOld = squeeze(PolicyDis.W.Dis(iH0,irho,ia,iphiP,:,:,:,:,:));
                    PolicyFinePOld = squeeze(PolicyDis.P.Dis(iH0,irho,ia,iphiP,:,:,:,:,:));
                    PolicyFineIOld = squeeze(double(PolicyDis.I.Dis(iH0,irho,ia,iphiP,:,:,:,:,:)));

                    PolicyMovFineE = squeeze(PolicyDis.E.Mov(iH0,irho,ia,iphiP,:,:,:,:));
                    PolicyMovFineA = squeeze(PolicyDis.A.Mov(iH0,irho,ia,iphiP,:,:,:,:));
                    PolicyMovFineW = squeeze(PolicyDis.W.Mov(iH0,irho,ia,iphiP,:,:,:,:));
                    PolicyMovFineP = squeeze(PolicyDis.P.Mov(iH0,irho,ia,iphiP,:,:,:,:));
                    PolAMovHelp = zeros(nE,nA,nW,nY,nD);
                    PolWMovHelp = zeros(nE,nA,nW,nY,nD);
                    PolPMovHelp = zeros(nE,nA,nW,nY,nD);
                    for iiHelp = nE:(-1):1
    %                     v1MovHelp(iiHelp,:,:,:,:) = squeeze(ValueFunctionDis.Mov.Dis(iH0,irho,ia,iphiP,:,:,:,:));
    %                     v1MovHelp(iiHelp,:,2,:,:) = 0.5*(v1MovHelp(iiHelp,:,1,:,:) + v1MovHelp(iiHelp,:,3,:,:));
                        PolAMovHelp(iiHelp,:,:,:,:) = squeeze(PolicyDis.A.Mov(iH0,irho,ia,iphiP,:,:,:,:));
                        PolWMovHelp(iiHelp,:,:,:,:) = squeeze(PolicyDis.W.Mov(iH0,irho,ia,iphiP,:,:,:,:));
                        PolPMovHelp(iiHelp,:,:,:,:) = squeeze(PolicyDis.P.Mov(iH0,irho,ia,iphiP,:,:,:,:));
                        PolAMovHelp(iiHelp,:,2,:,:) = 0.5*(PolAMovHelp(iiHelp,:,1,:,:)+PolAMovHelp(iiHelp,:,3,:,:));
                        PolWMovHelp(iiHelp,:,2,:,:) = 0.5*(PolWMovHelp(iiHelp,:,1,:,:)+PolWMovHelp(iiHelp,:,3,:,:));
                        PolPMovHelp(iiHelp,:,2,:,:) = 0.5*(PolPMovHelp(iiHelp,:,1,:,:)+PolPMovHelp(iiHelp,:,3,:,:));
                    end

                    PolicyStayFineA = squeeze(PolicyDis.A.Stay(iH0,irho,ia,iphiP,:,:,:,:,:));
                    PolicyStayFineW = squeeze(PolicyDis.W.Stay(iH0,irho,ia,iphiP,:,:,:,:,:));
                    PolicyStayFineP = squeeze(PolicyDis.P.Stay(iH0,irho,ia,iphiP,:,:,:,:,:));
                    PolicyStayFineA(:,:,2,:,:) = 0.5*(PolicyStayFineA(:,:,1,:,:)+PolicyStayFineA(:,:,3,:,:));
                    PolicyStayFineW(:,:,2,:,:) = 0.5*(PolicyStayFineW(:,:,1,:,:)+PolicyStayFineW(:,:,3,:,:));
                    PolicyStayFineP(:,:,2,:,:) = 0.5*(PolicyStayFineP(:,:,1,:,:)+PolicyStayFineP(:,:,3,:,:));
                    v0 = squeeze(ValueFunctionDis.v0(iH0,irho,ia,iphiP,:,:,:,:,:));
                    v0(:,:,2,:,:) = 0.5*v0(:,:,1,:,:)+ 0.5*v0(:,:,3,:,:);

                    minV0 = min(v0,[],'all');

                    Diff = 1;
                    iter = 1;
                    PolicyCheck = 0;

                    %%% Continuous Search
                    TotalFineSearchTime = tic;
                    while (Diff > 1e-5 && iter < 100)
                        TotalIterationTime = tic;

                        [MonotonicityA,MonotonicityW,MonotonicityY,MonotonicityD] = arrayfun(@TestMonotonicityV0,Egrid5,Agrid5,Wgrid5,Ygrid5,Dgrid5);
                        fprintf("v0: Mon A = %d | Mon W = %d | Mon Y = %d | Mon D = %d \n",...
                                all(MonotonicityA,'all'),all(MonotonicityW,'all'),all(MonotonicityY,'all'),all(MonotonicityD,'all'))

    %                     helpiter = 1;
    %                     while (any(MonotonicityA == 0,'all') || any(MonotonicityW == 0,'all')) && helpiter < 10
    %                         tic
    %                         v0 = MonotoniceA(v0);
    %                         v0 = MonotoniceW(v0);
    %                         [MonotonicityA,MonotonicityW,MonotonicityY,MonotonicityD] = arrayfun(@TestMonotonicityV0,Egrid5,Agrid5,Wgrid5,Ygrid5,Dgrid5);
    %                         toc
    %                         helpiter = helpiter + 1;
    %                         fprintf("Monotoniced v0: Mon A = %d | Mon W = %d | Mon Y = %d | Mon D = %d \n",...
    %                             all(MonotonicityA,'all'),all(MonotonicityW,'all'),all(MonotonicityY,'all'),all(MonotonicityD,'all'))
    %                     end

                        MovSearchTime = tic;
                        [v1MovHelp,PolAMovHelp,PolWMovHelp,PolPMovHelp,iterFineMov,GradNorm] = arrayfun(@OptiSearchMovFineGradSmoothed,...
                                                                                        Egrid5,Agrid5,Wgrid5,Ygrid5,Dgrid5,PolAMovHelp,PolWMovHelp,PolPMovHelp);

                        for iter5 = 1:10
                            v0MovHelp = v1MovHelp;
                            [v1MovHelp,PolAMovHelp,PolWMovHelp,PolPMovHelp] = arrayfun(@MonotoniceAMov,Egrid5,Agrid5,Wgrid5,Ygrid5,Dgrid5);
                            [v1MovHelp,PolAMovHelp,PolWMovHelp,PolPMovHelp] = arrayfun(@MonotoniceWMov,Egrid5,Agrid5,Wgrid5,Ygrid5,Dgrid5);
                            [v1MovHelp,PolAMovHelp,PolWMovHelp,PolPMovHelp] = arrayfun(@MonotoniceYMov,Egrid5,Agrid5,Wgrid5,Ygrid5,Dgrid5);
                            [v1MovHelp,PolAMovHelp,PolWMovHelp,PolPMovHelp] = arrayfun(@MonotoniceDMov,Egrid5,Agrid5,Wgrid5,Ygrid5,Dgrid5);
                            if max(abs(v1MovHelp(:)-v0MovHelp(:))) > 1e-6
                                break
                            end
                        end
                        [v1MovFine,PolicyMovFineE,PolicyMovFineA,PolicyMovFineW,PolicyMovFineP] = arrayfun(@ValueCompareMov,Agrid4,Wgrid4,Ygrid4,Dgrid4);
                        fprintf("Time for MovSearch  = %g seconds \n",toc(MovSearchTime))

                        StaySearchTime = tic;
                        [v1StayFine,PolicyStayFineA,PolicyStayFineW,PolicyStayFineP,iterFineStay,GradNormStay] = arrayfun(@OptiSearchStayFineGradSmoothed,Egrid5,Agrid5,Wgrid5,Ygrid5,Dgrid5,PolicyStayFineA,PolicyStayFineW,PolicyStayFineP);
                        for iter5 = 1:10
                            v0StayFine = v1StayFine;
                            [v1StayFine,PolicyStayFineA,PolicyStayFineW,PolicyStayFineP] = arrayfun(@MonotoniceAStay,Egrid5,Agrid5,Wgrid5,Ygrid5,Dgrid5);
                            [v1StayFine,PolicyStayFineA,PolicyStayFineW,PolicyStayFineP] = arrayfun(@MonotoniceWStay,Egrid5,Agrid5,Wgrid5,Ygrid5,Dgrid5);
                            [v1StayFine,PolicyStayFineA,PolicyStayFineW,PolicyStayFineP] = arrayfun(@MonotoniceYStay,Egrid5,Agrid5,Wgrid5,Ygrid5,Dgrid5);
                            [v1StayFine,PolicyStayFineA,PolicyStayFineW,PolicyStayFineP] = arrayfun(@MonotoniceDStay,Egrid5,Agrid5,Wgrid5,Ygrid5,Dgrid5);
                            if max(abs(v1StayFine(:)-v0StayFine(:))) > 1e-6
                                break
                            end
                        end
                        fprintf("Time for StaySearch  = %g seconds \n",toc(StaySearchTime))

                        [v1,PolicyFineE,PolicyFineA,PolicyFineW,PolicyFineP,PolicyFineI] = arrayfun(@ValueCompareFine,Egrid5,Agrid5,Wgrid5,Ygrid5,Dgrid5);
                        [MonotonicityA,MonotonicityW,MonotonicityY,MonotonicityD] = arrayfun(@TestMonotonicityV1,Egrid5,Agrid5,Wgrid5,Ygrid5,Dgrid5);

                         helpiter = 1;
                         while (any(MonotonicityA == 0,'all') || any(MonotonicityW == 0,'all')) && helpiter < 10
                            
                             fprintf("v1: Mon A = %d | Mon W = %d | Mon Y = %d | Mon D = %d \n",...
                                        all(MonotonicityA,'all'),all(MonotonicityW,'all'),all(MonotonicityY,'all'),all(MonotonicityD,'all'))

                             if mod(helpiter,2)==0
                                 [v1,PolicyFineE,PolicyFineA,PolicyFineW,PolicyFineP,PolicyFineP,...
                                     PolAMovHelp,PolWMovHelp,PolPMovHelp,...
                                     PolicyStayFineA,PolicyStayFineW,PolicyStayFineP] = arrayfun(@MonotoniceV1_A,Egrid5,Agrid5,Wgrid5,Ygrid5,Dgrid5);
                             else
                                 [v1,PolicyFineE,PolicyFineA,PolicyFineW,PolicyFineP,PolicyFineP,...
                                     PolAMovHelp,PolWMovHelp,PolPMovHelp,...
                                     PolicyStayFineA,PolicyStayFineW,PolicyStayFineP] = arrayfun(@MonotoniceV1_W,Egrid5,Agrid5,Wgrid5,Ygrid5,Dgrid5);
                             end
                             [MonotonicityA,MonotonicityW,MonotonicityY,MonotonicityD] = arrayfun(@TestMonotonicityV1,Egrid5,Agrid5,Wgrid5,Ygrid5,Dgrid5);
                             helpiter = helpiter + 1;
                         end

                         fprintf("v1: Mon A = %d | Mon W = %d | Mon Y = %d | Mon D = %d \n",...
                             all(MonotonicityA,'all'),all(MonotonicityW,'all'),all(MonotonicityY,'all'),all(MonotonicityD,'all'))


    %                     v2MovFine = v1MovFine;
    %                     v2StayFine = v1StayFine;
    %                    iter2 = 0;
    %                    Diff2 = 1;
    %                     while iter2 < 50 && Diff2 > 1e-7
    %                         v2MovHelp = arrayfun(@PolicyIterationMovFine,Egrid5,Agrid5,Wgrid5,Ygrid5,Dgrid5);
    %                         [v2MovFine,PolicyMovFineE,PolicyMovFineA,PolicyMovFineW,PolicyMovFineP] = arrayfun(@ValueCompareMovPI,Agrid4,Wgrid4,Ygrid4,Dgrid4);
    %                         
    %                         v2StayFine = arrayfun(@PolicyIterationStayFine,Egrid5,Agrid5,Wgrid5,Ygrid5,Dgrid5);
    %                         [v2,PolicyFineE,PolicyFineA,PolicyFineW,PolicyFineP,PolicyFineI] = arrayfun(@ValueComparePIFine,Egrid5,Agrid5,Wgrid5,Ygrid5,Dgrid5);
    %                         Diff2 = max(abs(v2 - v1),[],'all');
    %                         [MonotonicityA,MonotonicityW,MonotonicityY,MonotonicityD] = arrayfun(@TestMonotonicityV2,Egrid5,Agrid5,Wgrid5,Ygrid5,Dgrid5);
    %                         
    %                         if all(MonotonicityY,'all') && all(MonotonicityD,'all') && all(MonotonicityA,'all') && all(MonotonicityW,'all')
    %                             v1 = v2;
    %                             iter2 = iter2 + 1;
    %                             fprintf("iter2 = %d | Mon A = %d | Mon W = %d | Mon Y = %d | Mon D = %d \n",...
    %                                 iter2,all(MonotonicityA,'all'),all(MonotonicityW,'all'),all(MonotonicityY,'all'),all(MonotonicityD,'all'))
    %                         else
    %                             fprintf("iter2 = %d | Mon A = %d | Mon W = %d | Mon Y = %d | Mon D = %d \n",...
    %                                 iter2,all(MonotonicityA,'all'),all(MonotonicityW,'all'),all(MonotonicityY,'all'),all(MonotonicityD,'all'))
    %                             break
    %                         end
    %                     end
    %                     
    %                     helpiter = 1;
    %                     while (any(MonotonicityA == 0,'all') || any(MonotonicityW == 0,'all')) && helpiter < 10
    %                         tic
    %                         v1 = MonotoniceA(v1);
    %                         v1 = MonotoniceW(v1);
    %                         [MonotonicityA,MonotonicityW,MonotonicityY,MonotonicityD] = arrayfun(@TestMonotonicityV1,Egrid5,Agrid5,Wgrid5,Ygrid5,Dgrid5);
    %                         toc
    %                         helpiter = helpiter + 1;
    %                     end

                        Diff = max(abs(v1 - v0),[],'all');
                        fprintf("Time for Fine Search Iteration %d = %g seconds | Diff = %g \n",iter,toc(TotalIterationTime),Diff)

                        nIDiff = length(find(PolicyFineIOld ~= PolicyFineI));
                        nEDiff = length(find(PolicyFineEOld ~= PolicyFineE));
                        nADiff = length(find(abs(PolicyFineAOld - PolicyFineA)>FineEpsA2));
                        nWDiff = length(find(abs(PolicyFineWOld - PolicyFineW)>FineEpsW2));
                        nPDiff = length(find(abs(PolicyFinePOld - PolicyFineP)>FineEpsP2));
                        nDiff = length(find(abs(PolicyFineEOld(:) - PolicyFineE(:))> 0 | abs(PolicyFineIOld(:) - PolicyFineI(:))>0 | abs(PolicyFineAOld(:) - PolicyFineA(:))>FineEpsA2 | ...
                                            abs(PolicyFineWOld(:) - PolicyFineW(:))>FineEpsW2 | abs(PolicyFinePOld(:) - PolicyFineP(:))>FineEpsP2));

                        maxIDiff = max(abs(PolicyFineIOld - PolicyFineI),[],'all');
                        maxEDiff = max(abs(PolicyFineEOld - PolicyFineE),[],'all');
                        maxADiff = max(abs(PolicyFineAOld - PolicyFineA),[],'all');
                        maxWDiff = max(abs(PolicyFineWOld - PolicyFineW),[],'all');
                        maxPDiff = max(abs(PolicyFinePOld - PolicyFineP),[],'all');

                        fprintf("Number of changed policy: I = %d, E = %d, A = %d, W = %d, P = %d \n",nIDiff,nEDiff,nADiff,nWDiff,nPDiff)
                        fprintf("Max difference in policy: I = %g, E = %g, A = %g, W = %g, P = %g \n",maxIDiff,maxEDiff,maxADiff,maxWDiff,maxPDiff)

                        if (nDiff < 0.0001*nE*nA*nW*nY*nD && maxEDiff < 0.01 && maxADiff < FineEpsA3 && maxWDiff < FineEpsW3 && maxPDiff < FineEpsP3) || (nDiff < 1e-7*nE*nA*nW*nY*nD)
                            PolicyCheck = PolicyCheck + 1;
                            if Diff < 1e-2 && PolicyCheck > 0
                                Diff = 0;
                            end
                        else
                            PolicyCheck = 0;
                        end

                        v0 = v1;
                        PolicyFineEOld = PolicyFineE;
                        PolicyFineAOld = PolicyFineA;
                        PolicyFineWOld = PolicyFineW;
                        PolicyFinePOld = PolicyFineP;
                        PolicyFineIOld = PolicyFineI;
                        iter = iter + 1;
                    end
                    fprintf("TIME FOR TOTAL GRID SEARCH FINE = %g seconds \n",toc(TotalFineSearchTime))

                    Policy.E.Opt(iH0,irho,ia,iphiP,:,:,:,:,:) = gather(PolicyFineE);
                    Policy.A.Opt(iH0,irho,ia,iphiP,:,:,:,:,:) = gather(PolicyFineA);
                    Policy.W.Opt(iH0,irho,ia,iphiP,:,:,:,:,:) = gather(PolicyFineW);
                    Policy.P.Opt(iH0,irho,ia,iphiP,:,:,:,:,:) = gather(PolicyFineP);
                    Policy.I.Opt(iH0,irho,ia,iphiP,:,:,:,:,:) = int16(gather(PolicyFineI));

                    Policy.E.Stay(iH0,irho,ia,iphiP,:,:,:,:,:) = gather(Egrid(Egrid5));
                    Policy.A.Stay(iH0,irho,ia,iphiP,:,:,:,:,:) = gather(PolicyStayFineA);
                    Policy.W.Stay(iH0,irho,ia,iphiP,:,:,:,:,:) = gather(PolicyStayFineW);
                    Policy.P.Stay(iH0,irho,ia,iphiP,:,:,:,:,:) = gather(PolicyStayFineP);

                    Policy.E.Mov(iH0,irho,ia,iphiP,:,:,:,:) = gather(PolicyMovFineE);
                    Policy.A.Mov(iH0,irho,ia,iphiP,:,:,:,:) = gather(PolicyMovFineA);
                    Policy.W.Mov(iH0,irho,ia,iphiP,:,:,:,:) = gather(PolicyMovFineW);
                    Policy.P.Mov(iH0,irho,ia,iphiP,:,:,:,:) = gather(PolicyMovFineP);

                    ValueFunction.v0(iH0,irho,ia,iphiP,:,:,:,:,:) = gather(v0);
                    ValueFunction.Stay.Fine(iH0,irho,ia,iphiP,:,:,:,:,:) = gather(v1StayFine);
                    ValueFunction.Mov.Fine(iH0,irho,ia,iphiP,:,:,:,:) = gather(v1MovFine);

                    tic
                    [WealthInv,Consumption,Vul] = arrayfun(@AdditionalVariablesFine,Egrid5,Agrid5,Wgrid5,Ygrid5,Dgrid5);
                    Policy.c.Opt(iH0,irho,ia,iphiP,:,:,:,:,:) = gather(Consumption);
                    Policy.w.Opt(iH0,irho,ia,iphiP,:,:,:,:,:) = gather(WealthInv);
                    Policy.Vul.Opt(iH0,irho,ia,iphiP,:,:,:,:,:) = gather(Vul);

                    [WealthInv,Consumption,Vul] = arrayfun(@AdditionalVariablesMovFine,Agrid4,Wgrid4,Ygrid4,Dgrid4);
                    Policy.c.Mov(iH0,irho,ia,iphiP,:,:,:,:) = gather(Consumption);
                    Policy.w.Mov(iH0,irho,ia,iphiP,:,:,:,:) = gather(WealthInv);
                    Policy.Vul.Mov(iH0,irho,ia,iphiP,:,:,:,:) = gather(Vul);

                    [WealthInv,Consumption,Vul] = arrayfun(@AdditionalVariablesStayFine,Egrid5,Agrid5,Wgrid5,Ygrid5,Dgrid5);
                    Policy.c.Stay(iH0,irho,ia,iphiP,:,:,:,:,:) = gather(Consumption);
                    Policy.w.Stay(iH0,irho,ia,iphiP,:,:,:,:,:) = gather(WealthInv);
                    Policy.Vul.Stay(iH0,irho,ia,iphiP,:,:,:,:,:) = gather(Vul);

                    Policy.NetSavings.Opt(iH0,irho,ia,iphiP,:,:,:,:,:) = PolicyFineA - Agrid(Agrid5);
                    Policy.NetWealthChange.Opt(iH0,irho,ia,iphiP,:,:,:,:,:) = PolicyFineW - Wgrid(Wgrid5);

                    toc

                    fprintf("------------------------------------------------------------------------\n")
                    fprintf("------------------------------------------------------------------------\n")
                end
            end
        end
        save(strcat(SaveFolder,"\Results_continuous_",Version,".mat"),'ValueFunction','Policy','Grid','Para', '-v7.3')
    end
end

save(strcat(SaveFolder,"\Results_continuous_",Version,".mat"),'ValueFunction','Policy','Grid','Para', '-v7.3')
fprintf("Totaltime = %g seconds\n",toc(totaltime))
end
