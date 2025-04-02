
%robust/backtested portfolio optimization againts benchmark simple model by . k.Tomov

% Read the table of daily adjusted close prices for 2020-2024
T = readtable('Portfoliod.xlsx');
% Convert the table to a timetable and remove DJI stock
pricesTT = table2timetable(T(:,2:end));

% Handle missing price data using forward fill
pricesTT = fillmissing(pricesTT, 'previous', 'DataVariables', @isnumeric);

% Compute returns from prices
returnsTT = tick2ret(pricesTT);

% Define portfolio object with constraints
p = Portfolio;
p = estimateAssetMoments(p, returnsTT, 'MissingData', true, 'GetAssetList', true);
p = setBudget(p, 1, 1);       % Fully invested
p = setBounds(p, -0.2, 0.2); % Long-short bounds

% Define robustness parameter
kappa = 0.5;

% Set up backtest parameters
warmupPeriod = 21*2;          % 2-month warmup
rebalFreq = 21;               % Monthly rebalancing
lookback = [42 126];          % 2-6 month lookback windows
transactionCost = 0.005;      % 50bps transaction cost

% Calculate initial portfolios using warmup period
warmupTT = pricesTT(1:warmupPeriod,:);
initialMarkowitz = markowitzFcn([], warmupTT, p);
initialRP = robustFcn([], warmupTT, p, kappa);

% Plot initial allocations
bar([initialMarkowitz initialRP])
legend('Markowitz','Robust')
title('Initial Portfolio Allocation Comparison')

% Configure backtest strategies
stratMarkowitz = backtestStrategy('Markowitz', @(w,TT) markowitzFcn(w,TT,p), ...
    'RebalanceFrequency', rebalFreq, ...
    'LookbackWindow', lookback, ...
    'TransactionCosts', transactionCost, ...
    'InitialWeights', initialMarkowitz);

stratRobust = backtestStrategy('Robust', @(w,TT) robustFcn(w,TT,p,kappa), ...
    'RebalanceFrequency', rebalFreq, ...
    'LookbackWindow', lookback, ...
    'TransactionCosts', transactionCost, ...
    'InitialWeights', initialRP);

% Run backtest
backtester = backtestEngine([stratMarkowitz stratRobust]);
backtester = runBacktest(backtester, pricesTT, 'Start', warmupPeriod);

equityCurve(backtester)
summary(backtester)

% Portfolio construction functions
function new_weights = markowitzFcn(~, pricesTT, portObj)
    % Traditional mean-variance maximum return portfolio
    updatedPort = estimateAssetMoments(portObj, pricesTT, ...
        'DataFormat', 'Prices', 'MissingData', true);
    new_weights = estimateFrontierLimits(updatedPort, 'max');
end

function new_weights = robustFcn(~, pricesTT, portObj, kappa)
    % Robust portfolio with estimation error adjustment
    updatedPort = estimateAssetMoments(portObj, pricesTT, ...
        'DataFormat', 'Prices', 'MissingData', true);
    
    % Extract asset variances directly from covariance matrix
    assetVariances = diag(updatedPort.AssetCovar);
    
    % Define robust objective function using vector operations
    robustObjective = @(x) updatedPort.AssetMean'*x - ...
        kappa * sqrt(sum((x.^2) .* assetVariances));
    
    % Optimize portfolio weights
    new_weights = estimateCustomObjectivePortfolio(updatedPort, robustObjective, ...
        'ObjectiveSense', 'maximize');
end