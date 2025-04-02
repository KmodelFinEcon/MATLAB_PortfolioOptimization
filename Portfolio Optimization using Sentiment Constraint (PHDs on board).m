%% Portfolio Optimization using Social Performance Measure: Individuals with PhDs on Board
%% by K.Tomov
%% Data Setup

Assets = readtable("Logreturns") %>>>>> (Vector assets in excel file given daily logreturns)

%percentage of individual on board of each company in function vector of
%assets logretunrs

PhDs = [0.2857; 0.5; 0.3; 0.2857; 0.3077; 0.2727; ...
        0.4167; 0.2143; 0.3; 0.4167; 0.3077];

% PhD percentages view
table(PhDs, 'VariableNames', {'PhD_Percentage'}, 'RowNames', Assets(1:11))

%% Create and Configure the Portfolio


% Create portfolio with default constraints.
p = Portfolio('AssetList', Assets(1:11));
p = estimateAssetMoments(p, Data(:,1:11));
p = setDefaultConstraints(p);

% Define group constraints.G defines two groups and their lower
G = [0 1 0 0 1 0 0 0 0 0 0;
     1 0 0 1 0 0 1 0 0 0 0];
LowG = [0.15; 0.25];
UpG = [Inf; 0.5];
p = setGroups(p, G, LowG, UpG);

%% Define Objective Function

% average percentage of individuals with PhDs as the objective.
objectiveFunc = @(x) PhDs' * x;

% searching for portfolio that minimizes the average PhD percentage.
wgt_minPhD = estimateCustomObjectivePortfolio(p, objectiveFunc);
minPhD = objectiveFunc(wgt_minPhD);

% Finding the portfolio that maximizes the average PhD percentage.
wgt_maxPhD = estimateCustomObjectivePortfolio(p, objectiveFunc, ...
    "ObjectiveSense", "maximize");
maxPhD = objectiveFunc(wgt_maxPhD);
%% Create 3D Grid for Efficient Frontier

N = 20; % Number of grid points
targetPhD = linspace(minPhD, maxPhD, N);

%inequality constraint to enforce a minimum PhD percentage
Ain = -PhDs';
bin = -minPhD;
p = setInequality(p, Ain, bin);

% cells to store risk, return, and PhD percentages along the frontier
prsk = cell(N, 1);
pret = cell(N, 1);
pPhD = cell(N, 1);

for i = 1:N
    p.bInequality = -targetPhD(i);% Update the PhD constraint for the current target level
    pwgt = estimateFrontier(p, N);% Estimate the frontier weights
    [prsk{i}, pret{i}] = estimatePortMoments(p, pwgt);% Compute portfolio moments
    pPhD{i} = pwgt' * PhDs;% Compute the average PhD percentage
end

scatter3(cell2mat(prsk), cell2mat(pret), cell2mat(pPhD))
title('Efficient Portfolios')
xlabel('Risk Level')
ylabel('Expected Return')
zlabel('Percentage of PhDs')

%% Plot Contours


nC = 5; % Number of contour plots
minContour = max(pPhD{1}); % Lower bound for PhD percentage on overlapping contours

plotContours(p, minContour, maxPhD, nC, N)

%% Asset Exclusion Example based on PhD Threshold

% Remove the average PhD constraint.
p.AInequality = [];
p.bInequality = [];

thresholdPhD = 0.25:0.05:0.40;% Define a set of thresholds for excluding assets based on PhD percentage.

plotExclusionExample(p, PhDs, thresholdPhD, N)% Plot the efficient frontier for different exclusion thresholds.

%% Asset Exclusion and Impact on Return

% Exclude assets with a PhD percentage below 33%.
ub = zeros(p.NumAssets, 1);
ub(PhDs >= 0.33) = 1;
p.UpperBound = ub;

% portfolio for a given risk level after exclusion. in this case its 0.012
pwgt_exclude = estimateFrontierByRisk(p, 0.012);
ret_exclude = estimatePortReturn(p, pwgt_exclude);

p.UpperBound = [];% Return constraints to the original portfolio.

%% Incorporate Minimum PhD Constraint into Portfolio
p = addInequality(p, -PhDs', -0.33);

% The portfolio with the added minimum PhD constraint.
pwgt_avgPhD = estimateFrontierByRisk(p, 0.012);
ret_avgPhD = estimatePortReturn(p, pwgt_avgPhD);

% Remove the PhD inequality constraint.
p.AInequality = [];
p.bInequality = [];

ret_increase = (ret_avgPhD - ret_exclude) / ret_exclude;% Calculate return increase due to the average PhD constraint.

%% Global Function Definitions

function [] = plotContours(p, minPhD, maxPhD, nContour, nPort)    % Plot contour lines for different minimum PhD percentage levels.

    contourPhD = linspace(minPhD, maxPhD, nContour+1);    % Define a set of PhD percentage levels for contour plotting.
    
    figure;
    hold on
    labels = strings(nContour+1, 1);
    
    for i = 1:nContour
        p.bInequality = -contourPhD(i);
        plotFrontier(p, nPort);
        labels(i) = sprintf("%6.2f%% PhDs", contourPhD(i)*100);
    end
    
    p.AInequality = [];
    p.bInequality = [];
    plotFrontier(p, nPort);
    labels(nContour+1) = "No PhD restriction";
    
    legend(labels, 'Location', 'northwest')
    hold off
end

% Efficient frontier when excluding assets below varying PhD thresholds.

function [] = plotExclusionExample(p, PhDs, thresholdPhD, nPort)

    
    nT = length(thresholdPhD);
    figure;
    hold on
    labels = strings(nT+1, 1);
    
    for i = 1:nT
        ub = zeros(p.NumAssets, 1)
        ub(PhDs >= thresholdPhD(i)) = 1;% Only include assets with a PhD percentage above the threshold.
        p.UpperBound = ub;
        plotFrontier(p, nPort);
        labels(i) = sprintf("%6.2f%% PhDs", thresholdPhD(i)*100);
    end

    p.UpperBound = []; %frontier with no exclusions
    plotFrontier(p, nPort);
    labels(nT+1) = "No PhD restriction";
    
    legend(labels, 'Location', 'northwest')
    hold off
end