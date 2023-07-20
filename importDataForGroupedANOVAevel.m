%% Import data from spreadsheet
% Script for importing data from the following spreadsheet:
%
%    Workbook: C:\Users\Dominik\Documents\MATLAB\cas3dTool\CAS3dResultsSummary3.1.xls
%    Worksheet: CAS3dResultsSummary3.1
%
% Auto-generated by MATLAB on 01-Feb-2019 12:41:01

%% Setup the Import Options
opts = spreadsheetImportOptions("NumVariables", 9);

% Specify sheet and range
opts.Sheet = "CAS3dResultsSummary3.1";
opts.DataRange = "A2:I742";

% Specify column names and types
opts.VariableNames = ["Filename", "CAS3d", "intWeightedCAS3d", "MainDirection", "MainElevation", "WeightedAngularDeviation", "AngularDeviation", "Saturation", "Background"];
opts.VariableTypes = ["string", "string", "string", "string", "string", "string", "string", "string", "string"];
opts = setvaropts(opts, [1, 2, 3, 4, 5, 6, 7, 8, 9], "WhitespaceRule", "preserve");
opts = setvaropts(opts, [1, 2, 3, 4, 5, 6, 7, 8, 9], "EmptyFieldRule", "auto");

% Import the data
CAS3dResultsSummary1 = readtable("C:\Users\Dominik\Documents\MATLAB\cas3dTool\CAS3dResultsSummary3.1.xls", opts, "UseExcel", false);

%% Convert to output type
CAS3dResultsSummary1 = table2cell(CAS3dResultsSummary1);
numIdx = cellfun(@(x) ~isnan(str2double(x)), CAS3dResultsSummary1);
CAS3dResultsSummary1(numIdx) = cellfun(@(x) {str2double(x)}, CAS3dResultsSummary1(numIdx));

%% Clear temporary variables
clear  opts