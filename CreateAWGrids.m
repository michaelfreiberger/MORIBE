function [Agrid, Wgrid] = CreateAWGrids(lowA,highA,nA,highW,nW,Para)

lowA = max(Para.lowA,lowA);
lowA2 = sign(lowA)*abs(lowA)^(1/Para.exponentA);

highA2 = sign(highA)*abs(highA)^(1/Para.exponentA);
Agrid = sign(linspace(lowA2,highA2,nA)).*abs(linspace(lowA2,highA2,nA)).^Para.exponentA;
   
Wgrid = linspace( 0.0^(1/Para.exponentW), highW^(1/Para.exponentW), nW-1).^Para.exponentW;
Wgrid = sort([Wgrid,Para.lowW+1e-5]);


end

