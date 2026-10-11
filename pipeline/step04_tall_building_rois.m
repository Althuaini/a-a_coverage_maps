%% ============================================================
%  Building summary per ROI. For every downloaded city folder, reads the
%  OSM file of each ROI and writes:
%  - ROISummary.csv (ROIs with at least one building of 60 m or more)
%  - ROISkippedLog.csv (every skipped ROI and the reason)
%  - CityRunSummary_50City.csv (counts per city)
%  - Tall building threshold used here: MaxHeight >= 60
% ============================================================

%% === Settings ===
RESUME = true;              % true: skip ROIs already in ROISummary.csv
LOG_ALREADY_PROCESSED = false; % if RESUME=true, log "AlreadyProcessed" skips too?
SKIP_PREVIOUSLY_SKIPPED = true;

TALL_THRESH = 60;           % tall if MaxHeight >= 60

%% === Define folders ===
responseFolder  = '<DATA_ROOT>/01_Process/01.1_OverpassResponses/';
roiGridCSV      = "<MATLAB_DRIVE>/AIAA/bbox/All_Cities_ROI_Grid1.csv";

outputFolder    = '<DATA_ROOT>/01_Process/0.1_Buildings_summary_and_footprint/ROI_Summary_50City_TallBuildings_v02';
footprintFolder = fullfile(outputFolder, 'Footprints');

roiTable = readtable(roiGridCSV);

% Ensure output dirs exist
if ~exist(outputFolder, 'dir'), mkdir(outputFolder); end
if ~exist(footprintFolder, 'dir'), mkdir(footprintFolder); end

fprintf('\n=== Starting building summary across cities ===\n');

%% ============================================================
%  Output files
%% ============================================================
outCSV     = fullfile(outputFolder, 'ROISummary.csv');
skipCSV    = fullfile(outputFolder, 'ROISkippedLog.csv');
citySumCSV = fullfile(outputFolder, 'CityRunSummary_50City.csv');

%% ============================================================
%  Resume-safe processed ROI map
%% ============================================================
processedROIs = containers.Map(); % ROI -> true
skippedROIs   = containers.Map();

% --- load processed ROIs from summary CSV ---
if RESUME && isfile(outCSV)
    T = readtable(outCSV, "TextType","string");
    for k = 1:height(T)
        processedROIs(string(T.ROI(k))) = true;
    end
end

% --- load skipped ROIs from skip log CSV ---
if (RESUME && SKIP_PREVIOUSLY_SKIPPED) && isfile(skipCSV)
    S = readtable(skipCSV, "TextType","string");
    if any(strcmpi(S.Properties.VariableNames, "ROI"))
        for k = 1:height(S)
            r = string(S.ROI(k));
            if strlength(r) > 0
                skippedROIs(r) = true;
            end
        end
    end
end
%% ============================================================
%  Open summary CSV (append or write)
%% ============================================================
if isfile(outCSV)
    fid_out = fopen(outCSV, 'a');
else
    fid_out = fopen(outCSV, 'w');
    fprintf(fid_out, "City,ROI,BuildingsCount,Above60mCount\n");
end

%% ============================================================
%  Open skip log CSV (append or write)
%% ============================================================
if isfile(skipCSV)
    fid_skip = fopen(skipCSV, 'a');
else
    fid_skip = fopen(skipCSV, 'w');
    fprintf(fid_skip, "City,ROI,OSMFile,Reason,Detail\n");
end

% Ensure files close even if error happens
cleanupObj = onCleanup(@() closeFiles(fid_out, fid_skip));

%% ============================================================
%  Discover all city folders
%% ============================================================
cityFolders = dir(responseFolder);
cityFolders = cityFolders([cityFolders.isdir] & ~ismember({cityFolders.name},{'.','..'}));
cityNames   = {cityFolders.name};

% City-level summary table
citySummary = table('Size',[numel(cityNames) 4], ...
    'VariableTypes', ["string","double","double","double"], ...
    'VariableNames', ["City","OSM_Files","Processed_ROIs","Skipped_ROIs"]);

%% ============================================================
%  MAIN LOOP: Process each city
%% ============================================================
for c = 1:numel(cityNames)
    cityName = string(cityNames{c});
    fprintf('\n📍 Processing city: %s\n', cityName);

    cityPath = fullfile(responseFolder, cityName);
    osmFiles = dir(fullfile(cityPath, '*.osm'));

    processedCount = 0;
    skippedCount   = 0;

    citySummary.City(c) = cityName;
    citySummary.OSM_Files(c) = numel(osmFiles);

    if isempty(osmFiles)
        fprintf('  ⚠️  No OSM files found for %s\n', cityName);
        logSkip(fid_skip, cityName, "", "", "NoOSMFilesInCity", "City folder contains no *.osm files");
        citySummary.Processed_ROIs(c) = 0;
        citySummary.Skipped_ROIs(c)   = 0;
        continue;
    end

    for i = 1:numel(osmFiles)
        osmFileName = string(osmFiles(i).name);
        osmFilePath = fullfile(osmFiles(i).folder, osmFiles(i).name);

        roiID = "";
        reason = "";
        detail = "";

        % ---- Extract ROIID from filename ----
        parts = split(osmFileName, '_');
        roiIdx = find(contains(lower(parts),'roi'), 1);
        if isempty(roiIdx) || roiIdx+1 > numel(parts)
            skippedCount = skippedCount + 1;
            logSkip(fid_skip, cityName, "", osmFileName, "FilenameParseFailed", "Expected pattern *_roi_####.osm");
            continue;
        end
        roiID = erase(parts{roiIdx+1}, '.osm');
        roiID = string(roiID);

        % ---- Resume skip ----
        if RESUME && isKey(processedROIs, roiID)
            if LOG_ALREADY_PROCESSED
                skippedCount = skippedCount + 1;
                logSkip(fid_skip, cityName, roiID, osmFileName, "AlreadyProcessed", "ROI exists in ROISummary.csv");
            end
            continue;
        end

        % ---- Skip if previously skipped (from prior runs) ----
        if RESUME && SKIP_PREVIOUSLY_SKIPPED && isKey(skippedROIs, roiID)
            continue;
        end

        % ---- Read buildings + validate ----
        try
            bld = readgeotable(osmFilePath, 'Layer', 'buildings');
        catch ME
            skippedCount = skippedCount + 1;
            logSkip(fid_skip, cityName, roiID, osmFileName, "ReadGeoTableFailed", ME.message);
            continue;
        end

        if isempty(bld)
            skippedCount = skippedCount + 1;
            logSkip(fid_skip, cityName, roiID, osmFileName, "BuildingsLayerEmpty", "No buildings found in this ROI");
            continue;
        end

        if ~ismember('MaxHeight', bld.Properties.VariableNames)
            skippedCount = skippedCount + 1;
            logSkip(fid_skip, cityName, roiID, osmFileName, "MissingMaxHeight", "Buildings layer missing MaxHeight column");
            continue;
        end

        totalCount = height(bld);
        if totalCount == 0
            skippedCount = skippedCount + 1;
            logSkip(fid_skip, cityName, roiID, osmFileName, "TotalCountZero", "Buildings table has 0 rows");
            continue;
        end

        tallMask = bld.MaxHeight >= TALL_THRESH;
        over60Count = sum(tallMask);

        if over60Count == 0
            skippedCount = skippedCount + 1;
            logSkip(fid_skip, cityName, roiID, osmFileName, "NoTallBuildings", "MaxHeight>=60 count is 0");
            continue;
        end

        % ---- ROI bbox lookup (for stable plot bounds) ----
        rowIndex = str2double(roiID) + 1;
        if isnan(rowIndex) || rowIndex < 1 || rowIndex > height(roiTable)
            skippedCount = skippedCount + 1;
            logSkip(fid_skip, cityName, roiID, osmFileName, "ROIGridIndexOutOfBounds", ...
                sprintf("rowIndex=%g outside roiTable height=%d", rowIndex, height(roiTable)));
            continue;
        end

        minLat = roiTable.Min_Latitude(rowIndex);
        maxLat = roiTable.Max_Latitude(rowIndex);
        minLon = roiTable.Min_Longitude(rowIndex);
        maxLon = roiTable.Max_Longitude(rowIndex);

        % ---- Plot footprint ----
        try
            fig = figure('Visible','off');
            ax  = geoaxes(fig);

            geolimits(ax, [minLat maxLat], [minLon maxLon]);
            hold(ax,'on');

            shortMask = ~tallMask;

            if any(shortMask)
                geoplot(ax, bld(shortMask,:), "FaceColor", [0.2 0.45 0.85], "EdgeColor", "none");
            end
            if any(tallMask)
                geoplot(ax, bld(tallMask,:),  "FaceColor", [1.0 0.55 0.0], "EdgeColor", "none");
            end

            hold(ax,'off');

            title(ax, sprintf('%s – ROI %s | Buildings: %d | >=60m: %d', ...
                cityName, roiID, totalCount, over60Count), 'Interpreter','none');

            saveas(fig, fullfile(footprintFolder, sprintf('%s_%s_footprint.png', cityName, roiID)));

            drawnow limitrate;
            delete(fig);
        catch ME
            skippedCount = skippedCount + 1;
            logSkip(fid_skip, cityName, roiID, osmFileName, "PlotOrSaveFailed", ME.message);
            continue;
        end

        % ---- Write summary CSV row ----
        try
            fprintf(fid_out, '%s,%s,%d,%d\n', cityName, roiID, totalCount, over60Count);
            processedROIs(roiID) = true;
            processedCount = processedCount + 1;
        catch ME
            skippedCount = skippedCount + 1;
            logSkip(fid_skip, cityName, roiID, osmFileName, "CSVWriteFailed", ME.message);
            continue;
        end
    end

    citySummary.Processed_ROIs(c) = processedCount;
    citySummary.Skipped_ROIs(c)   = skippedCount;

    fprintf('  ✅ City done: processed=%d, skipped=%d, osmFiles=%d\n', ...
        processedCount, skippedCount, numel(osmFiles));
end

% Write city run summary
writetable(citySummary, citySumCSV);

fprintf('\n✅ DONE.\n');
fprintf('➡ Summary CSV: %s\n', outCSV);
fprintf('➡ Skip Log  : %s\n', skipCSV);
fprintf('➡ City Summary: %s\n\n', citySumCSV);


%% ===== helper functions =====
function logSkip(fid, city, roi, osmFile, reason, detail)
    % Escape commas/newlines in detail to keep CSV sane
    if strlength(detail) > 0
        detail = replace(string(detail), newline, " ");
        detail = replace(detail, ",", ";");
    end
    fprintf(fid, '%s,%s,%s,%s,%s\n', city, roi, osmFile, reason, detail);
end

function closeFiles(fid1, fid2)
    if fid1 > 0, fclose(fid1); end
    if fid2 > 0, fclose(fid2); end
end
