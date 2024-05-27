function [Egrid,Agrid,Wgrid,Dgrid,Pgrid,Ygrid_H0,TransY_H0,YTransfer_H0,H0grid,rhogrid,agrid,phiPgrid] = CreateGrids(nE,nA,nW,nY,nD,nP,Para)

Egrid = linspace( Para.lowE,Para.highE, nE);
Agrid = linspace( Para.lowA, Para.highA(1), nA);
Wgrid = linspace( 0.0, Para.highW(1), nW-1);
Wgrid = sort([Wgrid,Para.lowW+1e-5]);
Dgrid = linspace(0,1,nD);
Pgrid = linspace(0,1,nP);

if Para.nH0 == 1
    H0grid = (Para.lowH0 + Para.highH0)/2;
else
    H0grid = linspace(Para.lowH0, Para.highH0, Para.nH0);
end

if Para.nrho == 1
    rhogrid = (Para.lowrho + Para.highrho)/2;
else
    rhogrid = linspace(Para.lowrho, Para.highrho ,Para.nrho);
end

if Para.na == 1
    agrid = (Para.lowa + Para.higha)/2;
else
    agrid = linspace(Para.lowa, Para.higha ,Para.na);
end

if Para.nphiP == 1
    phiPgrid = (Para.lowphiP + Para.highphiP)/2;
else
    phiPgrid = linspace(Para.lowphiP,Para.highphiP,Para.nphiP);
end

Ygrid_H0 = zeros(Para.nH0,nY);
TransY_H0 = zeros(Para.nH0,nY,nY);
YTransfer_H0 = zeros(Para.nH0,nY);
for iH = 1:Para.nH0
    [Y,TransY_H0(iH,:,:)] = tauchen(nY,Para.Y.rho(iH),Para.Y.sigma(iH),Para.Y.lambda);
    Ygrid_H0(iH,:) = Y .* Para.Y.Ymean(iH);
    YTransfer_H0(iH,:) = max(0.0,Para.Ymin - Ygrid_H0(iH,:));
end

end