function [] = Household_Simulations_Paths(GPU,Filename,nSamp,T)
 
gpuDevice(GPU);

Results = load(strcat("D:\Users\mfreiber\DisasterRiskModel\Matlab-Simulations\",Filename));

dotPosition = 0;
for ii = 1:strlength(Filename)
    if Filename(ii) == "."
        dotPosition = ii;
        break
    end
end
VersionNumber = Filename(20:(dotPosition-1));

Para = Results.Para;
VF = Results.ValueFunction;
Policy = Results.Policy;
Grid = Results.Grid;

VF_Stay = VF.Stay.Fine;
VF_Mov = VF.Mov.Fine;

Emov = Policy.E.Mov;
Amov = Policy.A.Mov;
Astay = Policy.A.Stay;
Wmov = Policy.W.Mov;
Wstay = Policy.W.Stay;
Pmov = Policy.P.Mov;
Pstay = Policy.P.Stay;
cmov = Policy.c.Mov;
cstay = Policy.c.Stay;
wmov = Policy.w.Mov;
wstay = Policy.w.Stay;
Vulmov = Policy.Vul.Mov;
Vulstay = Policy.Vul.Stay;

%% Load equilibrium distribution

SaveFolder = strcat("D:\Users\mfreiber\Matlab-Simulations\VulnerabilityModel\HHSim",num2str(VersionNumber),"\Continuous");
filenames=dir(strcat(SaveFolder,'\*.mat'));
nFiles = 0;
for ii = 1:length(filenames)
    if strcmp(filenames(ii).name(1:4),'Part')
        nFiles = nFiles + 1;
    end
end

data = load(strcat(SaveFolder,"\",filenames(1).name));
Para = data.Para;
Grid = data.Grid;


%% Grid Definitions
Egrid = Grid.E;
Agrid = squeeze(Grid.A(1,1,1,1,:))';
Wgrid = Grid.W;
Dgrid = Grid.D;
Ygrid = Grid.Y;

nD = length(Dgrid);
nE = length(Egrid);
nA = length(Agrid);
nW = length(Wgrid);
nY = size(Ygrid,2);

H0grid = Grid.H0;
rhogrid = Grid.rho;
agrid = Grid.a;
phiPgrid = Grid.phiP;

nH0 = length(H0grid);
nrho = length(rhogrid);
na = length(agrid);
nphiP = length(phiPgrid);

TransY_H0 = Para.TransY;
      

%% ---------------------------------------------------------------------------------------------------

Emov(Emov==0) = Egrid(nE);
Amov(Emov==0) = Agrid(1);
Wmov(Emov==0) = Wgrid(1);
Pmov(Emov==0) = 0;

%% ---------------------------------------------------------------------------------------------------
function [Eprime,Aprime,Wprime,Pprime,Iprime,cprime,wprime,Vulprime,NetSavings,Dprime,Yprime,W2prime] = HHSim(iHH)
    %iH0,irho,ia,iphiP,
    E = HHPath(iH0,irho,ia,iphiP,iHH,time,1);
    A = HHPath(iH0,irho,ia,iphiP,iHH,time,2);
    W = HHPath(iH0,irho,ia,iphiP,iHH,time,3);
    Y = HHPath(iH0,irho,ia,iphiP,iHH,time,4);
    D = HHPath(iH0,irho,ia,iphiP,iHH,time,5);
    
    jE = 0;
    for kE = 1:nE
        if Egrid(kE) == E
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
        cprime = (1-tA)*((1-tW)*cstay(iH0,irho,ia,iphiP,jE, jA  ,jW  ,jY,jD,1)   ...
            +  tW *cstay(iH0,irho,ia,iphiP,jE, jA  ,jW+1,jY,jD,1))  ...
            + tA *((1-tW)*cstay(iH0,irho,ia,iphiP,jE, jA+1,jW  ,jY,jD,1)   ...
            +  tW *cstay(iH0,irho,ia,iphiP,jE, jA+1,jW+1,jY,jD,1));
        wprime = (1-tA)*((1-tW)*wstay(iH0,irho,ia,iphiP,jE, jA  ,jW  ,jY,jD,1)   ...
            +  tW *wstay(iH0,irho,ia,iphiP,jE, jA  ,jW+1,jY,jD,1))  ...
            + tA *((1-tW)*wstay(iH0,irho,ia,iphiP,jE, jA+1,jW  ,jY,jD,1)   ...
            +  tW *wstay(iH0,irho,ia,iphiP,jE, jA+1,jW+1,jY,jD,1));
        Vulprime = (1-tA)*((1-tW)*Vulstay(iH0,irho,ia,iphiP,jE, jA  ,jW  ,jY,jD,1)   ...
            +  tW *Vulstay(iH0,irho,ia,iphiP,jE, jA  ,jW+1,jY,jD,1))  ...
            + tA *((1-tW)*Vulstay(iH0,irho,ia,iphiP,jE, jA+1,jW  ,jY,jD,1)   ...
            +  tW *Vulstay(iH0,irho,ia,iphiP,jE, jA+1,jW+1,jY,jD,1));
        Iprime = 0;
    else
        if checkE == 1
            Eprime = Emov(iH0,irho,ia,iphiP,jA,jW,jY,jD);
        else
            if tA <= RelocateHelp(iHH,time,1) && tW <= RelocateHelp(iHH,time,2)
                Eprime = Emov(iH0,irho,ia,iphiP,jA,jW,jY,jD);
            elseif tA <= RelocateHelp(iHH,time,1) && tW <= RelocateHelp(iHH,time,2)
                Eprime = Emov(iH0,irho,ia,iphiP,jA,jW+1,jY,jD);
            elseif tA <= RelocateHelp(iHH,time,1) && tW <= RelocateHelp(iHH,time,2)
                Eprime = Emov(iH0,irho,ia,iphiP,jA+1,jW,jY,jD);
            else
                Eprime = Emov(iH0,irho,ia,iphiP,jA+1,jW+1,jY,jD);
            end
        end
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
        cprime = (1-tA)*((1-tW)*cmov(iH0,irho,ia,iphiP, jA  ,jW  ,jY,jD,1)   ...
            + tW *cmov(iH0,irho,ia,iphiP, jA  ,jW+1,jY,jD,1))  ...
            + tA *((1-tW)*cmov(iH0,irho,ia,iphiP, jA+1,jW  ,jY,jD,1)   ...
            + tW *cmov(iH0,irho,ia,iphiP, jA+1,jW+1,jY,jD,1));
        wprime = (1-tA)*((1-tW)*wmov(iH0,irho,ia,iphiP, jA  ,jW  ,jY,jD,1)   ...
            + tW *wmov(iH0,irho,ia,iphiP, jA  ,jW+1,jY,jD,1))  ...
            + tA *((1-tW)*wmov(iH0,irho,ia,iphiP, jA+1,jW  ,jY,jD,1)   ...
            + tW *wmov(iH0,irho,ia,iphiP, jA+1,jW+1,jY,jD,1));
        Vulprime = (1-tA)*((1-tW)*Vulmov(iH0,irho,ia,iphiP, jA  ,jW  ,jY,jD,1)   ...
            + tW *Vulmov(iH0,irho,ia,iphiP, jA  ,jW+1,jY,jD,1))  ...
            + tA *((1-tW)*Vulmov(iH0,irho,ia,iphiP, jA+1,jW  ,jY,jD,1)   ...
            + tW *Vulmov(iH0,irho,ia,iphiP, jA+1,jW+1,jY,jD,1));
        Iprime = 1;
    end
    
    NetSavings = Aprime - A;
    
    jYprime = 1;
    TransY_Cum = TransY_H0(iH0,jY,1);
    for kY = 1:nY
        if kY == nY
            jYprime = nY;
        elseif Ysim(iHH,time) >= TransY_Cum
            TransY_Cum = TransY_Cum + TransY_H0(iH0,jY,kY + 1);
            jYprime = jYprime + 1;
        else
            break
        end
    end
    Yprime = Ygrid(iH0,jYprime);
    
    if Dsim(iHH,time) < Eprime
        Dprime = 1;
    else
        Dprime = 0;
    end
    
    if Dprime == 1
        W2prime = Wprime * Pprime;
    else
        W2prime = Wprime;
    end    
end

nHHGridSim = gpuArray(1:nSamp);
%nHHGridSim = 1:nSamp;

HHPath = zeros(nH0,nrho,na,nphiP,nSamp,T,11);

Dsim = rand(nSamp,T);
Ysim = rand(nSamp,T);
RelocateHelp = rand(nSamp,T,2);


SaveFolder = strcat("D:\Users\mfreiber\Matlab-Simulations\VulnerabilityModel\HHSim",num2str(VersionNumber),"\Continuous");
mkdir(SaveFolder);

for iH0 = 1:nH0
    for irho = 1:nrho
        for ia = 1:na
            for iphiP = 1:nphiP
                tic
                EquData = load(strcat(SaveFolder,"\Part_",num2str(iH0),"_",num2str(irho),"_",num2str(ia),"_",num2str(iphiP),".mat"));
                
                Agrid = squeeze(Grid.A(iH0,irho,ia,iphiP,:))';
                nHH = length(EquData.Dec.E);
                
                HHsamp = randsample(nHH,nSamp);
                
                HHPath(iH0,irho,ia,iphiP,:,1,1) = EquData.Path.E(HHsamp);
                HHPath(iH0,irho,ia,iphiP,:,1,2) = EquData.Path.A(HHsamp);
                HHPath(iH0,irho,ia,iphiP,:,1,3) = EquData.Path.W(HHsamp);
                HHPath(iH0,irho,ia,iphiP,:,1,4) = EquData.Path.Y(HHsamp);
                HHPath(iH0,irho,ia,iphiP,:,1,5) = EquData.Path.D(HHsamp);
                                               
                for time = 1:T
                    [Eprime,Aprime,Wprime,Pprime,Iprime,cprime,wprime,Vulprime,NetSavings,Dprime,Yprime,W2prime] = arrayfun(@HHSim,nHHGridSim);
                    HHPath(iH0,irho,ia,iphiP,:,time,6) = Iprime;
                    HHPath(iH0,irho,ia,iphiP,:,time,7) = Pprime;
                    HHPath(iH0,irho,ia,iphiP,:,time,8) = cprime;
                    HHPath(iH0,irho,ia,iphiP,:,time,9) = wprime;
                    HHPath(iH0,irho,ia,iphiP,:,time,10) = NetSavings;
                    HHPath(iH0,irho,ia,iphiP,:,time,11) = Vulprime;
                    if time < T
                        HHPath(iH0,irho,ia,iphiP,:,time+1,1) = Eprime;
                        HHPath(iH0,irho,ia,iphiP,:,time+1,2) = Aprime;
                        HHPath(iH0,irho,ia,iphiP,:,time+1,3) = W2prime;
                        HHPath(iH0,irho,ia,iphiP,:,time+1,4) = Yprime;
                        HHPath(iH0,irho,ia,iphiP,:,time+1,5) = Dprime;
                    end
                end
                toc
            end
        end
    end
end
save(strcat(SaveFolder,"\HHPaths.mat"),'HHPath','-v7.3')


end
