%Stock selection optimization algorithm in function of implied risk.

T = readtable("purchasewithSNPnodate.xlsx")
[nObservation, p] = size(T)
splitPoint = ceil(nObservation*0.6);
training = 1:splitPoint;
test = splitPoint+1:nObservation;
trainingNum = numel(training);

plot(T{:,1:end});
x=('Timestep');
y=('Value');
title('Equity Curve');
legend(T.Properties.VariableNames(1:end), 'Location',"bestoutside",'Interpreter','none');
k = 2;
[factorLoading,factorRetn,latent,tsq,explained,mu] = pca(T{training,(1:end)}, 'NumComponents', k);

dbstop if error

[denoisedCov,numFactorsDenoising] = covarianceDenoising(T{training,(1:end)});
disp(numFactorsDenoising)
clear cov

covarFactor = cov(factorRetn);

reconReturn = factorRetn*factorLoading' + mu;
unexplainedRetn = T{training,(1:end)} - reconReturn;

unexplainedCovar = diag(cov(unexplainedRetn));
D = diag(unexplainedCovar);

targetRisk = 0.05;  % Standard deviation of portfolio return
tRisk = targetRisk*targetRisk;  % Variance of portfolio return
meanStockRetn = mean(T{training,(1:end)});

optimProb = optimproblem('Description','Portfolio with factor covariance matrix','ObjectiveSense','max');
wgtAsset = optimvar('asset_weight', p, 1, 'Type', 'continuous', 'LowerBound', 0, 'UpperBound', 1);
wgtFactor = optimvar('factor_weight', k, 1, 'Type', 'continuous');

optimProb.Objective = sum(meanStockRetn'.*wgtAsset);

optimProb.Constraints.asset_factor_weight = factorLoading'*wgtAsset - wgtFactor == 0;
optimProb.Constraints.risk = wgtFactor'*covarFactor*wgtFactor + wgtAsset'*D*wgtAsset <= tRisk;
optimProb.Constraints.budget = sum(wgtAsset) == 1;

x0.asset_weight = ones(p, 1)/p;
x0.factor_weight = zeros(k, 1);
opt = optimoptions("fmincon", "Algorithm","sqp", "Display", "off", ...
    'ConstraintTolerance', 1.0e-8, 'OptimalityTolerance', 1.0e-8, 'StepTolerance', 1.0e-8);
x = solve(optimProb,x0, "Options",opt);
assetWgt1 = x.asset_weight;

%Optimal balancing based on the optimoptions algorithm >

percentage = 0.15;
AssetName = T.Properties.VariableNames(assetWgt1>=percentage)';
Weight = assetWgt1(assetWgt1>=percentage);
T1 = table(AssetName, Weight)