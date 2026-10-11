% ========================================================================
% Script: step07_matlab_building_heights.m
%
% Purpose:
%   Extracts building height information from OSM (Overpass API) responses
%   for the cities with TX files in the TX folder (txrxBaseFolder below).
%
%   Each city folder in the TX folder (txrxBaseFolder)
%   contains multiple Tx-Rx pair CSVs, each corresponding to a Region of
%   Interest (ROI). ROI IDs are extracted from those filenames to filter
%   which OSM files are processed.
%
%   Only OSM files matching those ROI IDs are read from the OSM folder
%   (responseFolder).
%
%   Each processed file outputs a CSV with:
%       BuildingID, MaxHeight
%
%   Only ROIs with TX files are processed; these are the ROIs with at least
%   one building of 60 m or more.
%
% Output:
%   For each city: <outputFolder>/<City>/
%   Each CSV → "<OSMFile>_building_heights.csv"
%
%   Error and warning logs saved in "error_log.txt" in the same output dir.
%
% ========================================================================

%% === Define folders ===
responseFolder = '<DATA_ROOT>/01_Process/01.1_OverpassResponses/';
txrxBaseFolder = '<DATA_ROOT>/01_Process/01.2_txrx_generation/Tx_Data200feet/';
outputFolder   = '<DATA_ROOT>/01_Process/01.4_Building_Matrix/20City/Buildings_Height_ref_20City';

% Ensure output folder exists before opening log file
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Create error log file
errorLogFile = fullfile(outputFolder, 'error_log.txt');
fid = fopen(errorLogFile, 'w');
if fid == -1
    error('Unable to create log file: %s', errorLogFile);
end
fprintf(fid, 'Error log for Building Height Extraction\n===========================\n\n');

%% === Get list of 20 target cities from Tx_Data200feet ===
txrxCityFolders = dir(txrxBaseFolder);
txrxCityFolders = txrxCityFolders([txrxCityFolders.isdir] & ~ismember({txrxCityFolders.name}, {'.', '..'}));

% Clean up names by removing "_txrx_Gen" suffix if present
rawCityNames = {txrxCityFolders.name};
targetCities = regexprep(rawCityNames, '_txrx_Gen$', '');

fprintf('Found %d city folders in Tx_Data200feet\n', numel(targetCities));
disp('Cities to process:');
disp(targetCities');

%% === Loop over each city ===
for cityIdx = 1:length(targetCities)
    cleanCityName = targetCities{cityIdx};         % e.g. "Abu Dhabi_ROI"
    rawCityName   = txrxCityFolders(cityIdx).name; % e.g. "Abu Dhabi_ROI_txrx_Gen"

    fprintf('\nProcessing city: %s\n', cleanCityName);

    % Define paths
    cityResponsePath = fullfile(responseFolder, cleanCityName);  % OSM folder
    cityTxrxPath     = fullfile(txrxBaseFolder, rawCityName);    % TxRx CSV folder
    cityOutputFolder = fullfile(outputFolder, cleanCityName);

    % Validate folders
    if ~exist(cityResponsePath, 'dir')
        fprintf(fid, 'Warning: Missing Overpass folder for %s\n', cleanCityName);
        continue;
    end
    if ~exist(cityOutputFolder, 'dir')
        mkdir(cityOutputFolder);
    end

    %% === STEP 1: Extract ROI IDs from TxRx CSVs ===
    csvFiles = dir(fullfile(cityTxrxPath, '*_tx_rx_pairs.csv'));
    if isempty(csvFiles)
        fprintf(fid, 'Warning: No TxRx CSVs found for city %s\n', cleanCityName);
        continue;
    end

    roiIDs = strings(length(csvFiles),1);
    for k = 1:length(csvFiles)
        nameParts = split(csvFiles(k).name, '_');
        roiIdx = find(contains(nameParts, 'roi'), 1);
        if ~isempty(roiIdx) && roiIdx+1 <= length(nameParts)
            roiIDs(k) = nameParts{roiIdx+1};
        end
    end
    roiIDs = unique(roiIDs(~strcmp(roiIDs,"")));
    fprintf('City %s: %d ROI IDs from TxRx CSVs\n', cleanCityName, numel(roiIDs));

    %% === STEP 2: Find matching OSM files ===
    osmFiles = dir(fullfile(cityResponsePath, '*.osm'));
    matchMask = false(numel(osmFiles), 1);
    for i = 1:numel(osmFiles)
        if any(contains(osmFiles(i).name, roiIDs))
            matchMask(i) = true;
        end
    end
    matchedFiles = osmFiles(matchMask);
    fprintf('Matched %d OSM files for %s\n', numel(matchedFiles), cleanCityName);

    %% === STEP 3: Extract building heights ===
    for i = 1:length(matchedFiles)
        osmFilePath = fullfile(matchedFiles(i).folder, matchedFiles(i).name);
        try
            cityData = readgeotable(osmFilePath, 'Layer', 'buildings');
            if isempty(cityData)
                continue;
            end

            if ismember('MaxHeight', cityData.Properties.VariableNames)
                maxBuildingHeight = max(cityData.MaxHeight);
                if maxBuildingHeight < 60
                    continue;
                end
            else
                fprintf(fid, 'Warning: MaxHeight missing in %s\n', osmFilePath);
                continue;
            end

            % Create height-only table
            buildingIDs = cityData.ID;
            heights     = cityData.MaxHeight;
            heightTable = table(buildingIDs, heights, ...
                                'VariableNames', {'BuildingID', 'MaxHeight'});

            % Save output
            [~, baseFileName, ~] = fileparts(matchedFiles(i).name);
            outputCSV = fullfile(cityOutputFolder, ...
                [baseFileName '_building_heights.csv']);
            writetable(heightTable, outputCSV);

        catch ME
            fprintf(fid, 'Error in %s: %s\n', osmFilePath, ME.message);
        end
    end
end

%% === Wrap up ===
fclose(fid);
fprintf('All processing complete. Log saved: %s\n', errorLogFile);
