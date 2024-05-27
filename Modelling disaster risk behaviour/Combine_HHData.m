function [DATA] = Combine_HHData(varargin)

for jj = 1:2:nargin
    if strcmp('Version', varargin{jj})
        Version = varargin{jj+1};
    elseif strcmp('Type', varargin{jj})
        Type = varargin{jj+1};
    end
end

SaveFolder = strcat("D:\Users\mfreiber\DisasterRiskModel\Matlab-Simulations_V2.1\",Version,"\HHSim",Type);

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

nHH = length(data.Dec.E);
DATA = zeros(nFiles*nHH,26);

for ii = 1:length(filenames)
    tic
    fprintf(filenames(ii).name)
    if strcmp(filenames(ii).name(1:4),'Part')
        data = load(strcat(SaveFolder,"\",filenames(ii).name));
        % HHID
        DATA((1+(ii-1)*nHH):(ii*nHH),1) = (1+(ii-1)*nHH):(ii*nHH);
        % H0
        DATA((1+(ii-1)*nHH):(ii*nHH),2) = data.Grid.H0(str2double(filenames(ii).name(6)));
        % rho
        DATA((1+(ii-1)*nHH):(ii*nHH),3) = data.Grid.rho(str2double(filenames(ii).name(8)));
        % a
        DATA((1+(ii-1)*nHH):(ii*nHH),4) = data.Grid.a(str2double(filenames(ii).name(10)));
        % phiP
        DATA((1+(ii-1)*nHH):(ii*nHH),5) = data.Grid.phiP(str2double(filenames(ii).name(12)));
        % Class
        DATA((1+(ii-1)*nHH):(ii*nHH),6) = (str2double(filenames(ii).name(6))-1)*27 + (str2double(filenames(ii).name(8))-1)*9 + ...
                                          (str2double(filenames(ii).name(10))-1)*3 + str2double(filenames(ii).name(12));
        % E0                              
        DATA((1+(ii-1)*nHH):(ii*nHH),7) = data.Path.E;
        % A0
        DATA((1+(ii-1)*nHH):(ii*nHH),8) = data.Path.A;
        % W0
        DATA((1+(ii-1)*nHH):(ii*nHH),9) = data.Path.W;
        % Y0
        DATA((1+(ii-1)*nHH):(ii*nHH),10) = data.Path.Y;
        % D0
        DATA((1+(ii-1)*nHH):(ii*nHH),11) = data.Path.D;
        % E
        DATA((1+(ii-1)*nHH):(ii*nHH),12) = data.Dec.E;
        % A
        DATA((1+(ii-1)*nHH):(ii*nHH),13) = data.Dec.A;
        % W
        DATA((1+(ii-1)*nHH):(ii*nHH),14) = data.Dec.W;
        % P
        DATA((1+(ii-1)*nHH):(ii*nHH),15) = data.Dec.P;
        % I
        DATA((1+(ii-1)*nHH):(ii*nHH),16) = data.Dec.I;
        % c
        DATA((1+(ii-1)*nHH):(ii*nHH),17) = data.Dec.c;
        % Net-Savings
        DATA((1+(ii-1)*nHH):(ii*nHH),18) = data.Dec.NS;
        % w
        DATA((1+(ii-1)*nHH):(ii*nHH),19) = data.Dec.w;
        % Vul
        DATA((1+(ii-1)*nHH):(ii*nHH),20) = data.Dec.Vul;
        % pESubsidy
        DATA((1+(ii-1)*nHH):(ii*nHH),21) = data.Policies.pESubsidy;
        % YTransfer
        DATA((1+(ii-1)*nHH):(ii*nHH),22) = data.Policies.YTransfer;
        % pETransfer
        DATA((1+(ii-1)*nHH):(ii*nHH),23) = data.Policies.pETransfer;
        % PrevExpSubsidy
        DATA((1+(ii-1)*nHH):(ii*nHH),24) = data.Policies.PrevExpSubsidy;
        % ExpU1
        DATA((1+(ii-1)*nHH):(ii*nHH),25) = data.HHUtility(:,1);
        % ExpU2
        DATA((1+(ii-1)*nHH):(ii*nHH),26) = data.HHUtility(:,2);
    end
    toc
end

Grid.pE = gather(Grid.pE);
save(strcat(SaveFolder,"\TOTALDATA.mat"),'DATA','Grid','Para', '-v7.3')
end

