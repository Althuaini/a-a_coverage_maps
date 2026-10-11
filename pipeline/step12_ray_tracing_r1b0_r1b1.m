%% ===========================================================
% Ray-traced coverage maps R1B0 and R1B1 for the ROIs listed in
% df_20city_rois.csv (SBR, one reflection).
%   - Phase 1 (R1B0, terrain only): ROIs in parallel with parfor
%     (20 workers, GPU off in the workers)
%   - Phase 2 (R1B1, buildings and terrain): serial, GPU on; the site
%     viewer is loaded once per ROI and used for all 50 TX
%   - The ROI list is built before any parallel work
%   - Each worker writes its own log file; these are merged after the parfor
%   - Files that already exist are skipped, so a run can be resumed
% ===========================================================

%% === 1) Setup ===
gpu = gpuDevice;
disp(gpu.Name + " GPU detected and available.")

% Folder with TX CSV files
txFolder = '<DATA_ROOT>/01_Process/01.2_txrx_generation/Tx_Data200feet/';

% Get all CSV files from all subfolders
allCsvFiles = dir(fullfile(txFolder, '**', '*.csv'));

% Filter only those with the '_tx_rx_pairs' pattern
txFiles = allCsvFiles(contains({allCsvFiles.name}, '_tx_rx_pairs'));

% Folder with OSM files
osmFolder = '<DATA_ROOT>/01_Process/01.1_OverpassResponses/';

% Grid resolution — single source of truth (change here only)
gridRes = 100;

% Folder to save .h5 results
outputFolder = sprintf('<DATA_ROOT>/01_Process/01.3_Coverage_Maps/20City_Res%d/', gridRes);

% Create output folder if needed
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Log file for timings
logFile = fullfile(outputFolder, 'GPU_Coverage_timing_log.csv');
if ~isfile(logFile)
    fid = fopen(logFile, 'w');
    fprintf(fid, 'City,ROI,Tx_Index,RB,Time\n');
    fclose(fid);
end

% Load ROI bounding boxes
bboxData = readtable('All_Cities_ROI_Grid1.csv');

% === Load df whitelist ===
dfPath = '<DATA_ROOT>/01_Process/01.3_Coverage_Maps/df_20city_rois.csv';
df = readtable(dfPath, 'TextType', 'string');
df.roi_id = string(df.roi_id);
df.city   = string(df.city);
fprintf('Loaded df whitelist: %d ROIs across %d cities.\n', ...
    height(df), numel(unique(df.city)));

% One reflection. R0B0 and R0B1 are computed in Python.
R = 1;

% RX grid
[Xs, Ys] = meshgrid(linspace(-500, 500, gridRes));
xlist = Xs(:); ylist = Ys(:);

%% === 2) Build roiList upfront — serial, just parsing filenames ===

roiList = struct('fileIdx',{}, 'roiID',{}, 'roiIDStr',{}, ...
                 'cityName',{}, 'txFileName',{});

for fileIdx = 1:length(txFiles)
    tokens = regexp(txFiles(fileIdx).name, 'roi_(\d+)_tx_rx_pairs', 'tokens');
    if isempty(tokens), continue; end

    roiIDStr_ = string(tokens{1}{1});

    [~, txFileBase, ~] = fileparts(txFiles(fileIdx).name);
    cityTokens = regexp(txFileBase, '^(.*?)_roi', 'tokens');
    if isempty(cityTokens), continue; end

    cityName_ = string(cityTokens{1}{1});

    if ~any(df.city == cityName_ & df.roi_id == roiIDStr_)
        continue;
    end

    entry.fileIdx    = fileIdx;
    entry.roiID      = str2double(roiIDStr_);
    entry.roiIDStr   = roiIDStr_;
    entry.cityName   = cityName_;
    entry.txFileName = fullfile(txFiles(fileIdx).folder, txFiles(fileIdx).name);
    roiList(end+1)   = entry;
end

fprintf('Total ROIs to process: %d\n', length(roiList));

%% === 3) Start parallel pool ===

if isempty(gcp('nocreate'))
    parpool('local', 20);
end

%% === PHASE 1: R1B0 — parfor across ROIs (CPU only, no siteviewer) ===

fprintf('\n=== PHASE 1: R1B0 (parallel across ROIs, GPU off per worker) ===\n');

% Explicit broadcast copies — parfor workers cannot access workspace vars
% by reference, only by value (sliced or broadcast)
bboxData_p   = bboxData;
outputFolder_p = outputFolder;
xlist_p      = xlist;
ylist_p      = ylist;
XsSize       = size(Xs);

parfor i = 1:length(roiList)

    roi      = roiList(i);
    roiID    = roi.roiID;
    cityName = roi.cityName;

    cityOutputFolder = fullfile(outputFolder_p, cityName);
    h5_R1B0 = fullfile(cityOutputFolder, ...
        sprintf('Output_%s_roi_%d_R1B0.h5', cityName, roiID));

    if isfile(h5_R1B0)
        fprintf('⏭️  R1B0 exists: ROI %d (%s)\n', roiID, cityName);
        continue;
    end

    roiIdx = roiID + 1;
    if roiIdx < 1 || roiIdx > height(bboxData_p)
        fprintf('Invalid ROI index: %d\n', roiIdx);
        continue;
    end

    tx_data = readtable(roi.txFileName);
    numTransmitters = height(tx_data);
    if numTransmitters < 1, continue; end

    minLat    = bboxData_p.Min_Latitude(roiIdx);
    maxLat    = bboxData_p.Max_Latitude(roiIdx);
    minLon    = bboxData_p.Min_Longitude(roiIdx);
    maxLon    = bboxData_p.Max_Longitude(roiIdx);
    centerLat = (minLat + maxLat) / 2;
    centerLon = (minLon + maxLon) / 2;
    [lats_w, lons_w] = xy2ll(xlist_p, ylist_p, centerLat, centerLon);

    if ~exist(cityOutputFolder, 'dir')
        mkdir(cityOutputFolder);
    end

    ssResults = cell(numTransmitters, 1);
    timings   = zeros(numTransmitters, 1);

    % Serial TX loop inside the worker — one core owns this entire ROI
    for txIdx = 1:numTransmitters
        txlat    = tx_data.txLat(txIdx);
        txlon    = tx_data.txLon(txIdx);
        txHeight = tx_data.txHeight(txIdx);

        tx_ = txsite('Latitude', txlat, 'Longitude', txlon, 'AntennaHeight', txHeight);
        rx_ = rxsite('Latitude', lats_w, 'Longitude', lons_w, 'AntennaHeight', txHeight);

        % UseGPU = off — each worker is a separate MATLAB session;
        % GPU context cannot be shared across parallel workers safely
        rtpm_ = propagationModel("raytracing", Method="sbr", ...
            MaxNumReflections=1, BuildingsMaterial="auto");
        rtpm_.UseGPU = "off";

        t = tic;
        ss = sigstrength(rx_, tx_, rtpm_);
        timings(txIdx) = toc(t);

        ss(~isfinite(ss)) = -120;
        ssResults{txIdx} = ss;
    end

    % HDF5 write — each worker writes its own file, no conflicts
    for txIdx = 1:numTransmitters
        gridSS = flipud(reshape(ssResults{txIdx}, XsSize));
        dset   = sprintf('/Tx_%d', txIdx);
        fileExists_ = isfile(h5_R1B0);
        if fileExists_
            info_          = h5info(h5_R1B0);
            existingDsets_ = {info_.Datasets.Name};
        else
            existingDsets_ = {};
        end
        if ~fileExists_ || ~any(strcmp(existingDsets_, dset))
            h5create(h5_R1B0, dset, size(gridSS));
        end
        h5write(h5_R1B0, dset, gridSS);
    end

    % Per-ROI temp log — avoids concurrent appends to shared logFile
    tempLog = fullfile(cityOutputFolder, sprintf('log_roi_%d_R1B0.tmp', roiID));
    fid = fopen(tempLog, 'w');
    for txIdx = 1:numTransmitters
        fprintf(fid, '%s,%d,%d,R1B0,%.4f\n', cityName, roiID, txIdx, timings(txIdx));
    end
    fclose(fid);

    fprintf('✅ R1B0 done: ROI %d (%s)\n', roiID, cityName);
end

% Merge temp logs into main log — serial, safe after parfor
fprintf('\nMerging Phase 1 temp logs...\n');
tmpLogs = dir(fullfile(outputFolder, '**', 'log_roi_*_R1B0.tmp'));
for k = 1:length(tmpLogs)
    tmpPath = fullfile(tmpLogs(k).folder, tmpLogs(k).name);
    content = fileread(tmpPath);
    fid = fopen(logFile, 'a');
    fprintf(fid, '%s', content);
    fclose(fid);
    delete(tmpPath);
end
fprintf('Phase 1 log merge complete — %d files merged.\n', length(tmpLogs));

%% === PHASE 2: R1B1 — serial, GPU on, siteviewer shared across all 50 TX ===

fprintf('\n=== PHASE 2: R1B1 (serial — GPU on, siteviewer per ROI) ===\n');

for i = 1:length(roiList)

    roi      = roiList(i);
    roiID    = roi.roiID;
    cityName = roi.cityName;

    try
        cityOutputFolder = fullfile(outputFolder, cityName);
        h5_R1B1 = fullfile(cityOutputFolder, ...
            sprintf('Output_%s_roi_%d_R1B1.h5', cityName, roiID));

        if isfile(h5_R1B1)
            fprintf('⏭️  R1B1 exists: ROI %d (%s)\n', roiID, cityName);
            continue;
        end

        fprintf('\n=== R1B1 | City: %s | ROI: %d ===\n', cityName, roiID);

        roiIdx = roiID + 1;
        if roiIdx < 1 || roiIdx > height(bboxData)
            fprintf('Invalid ROI index: %d\n', roiIdx);
            continue;
        end

        tx_data = readtable(roi.txFileName);
        numTransmitters = height(tx_data);
        if numTransmitters < 1, continue; end

        minLat    = bboxData.Min_Latitude(roiIdx);
        maxLat    = bboxData.Max_Latitude(roiIdx);
        minLon    = bboxData.Min_Longitude(roiIdx);
        maxLon    = bboxData.Max_Longitude(roiIdx);
        centerLat = (minLat + maxLat) / 2;
        centerLon = (minLon + maxLon) / 2;
        [lats, lons] = xy2ll(xlist, ylist, centerLat, centerLon);

        if ~exist(cityOutputFolder, 'dir')
            mkdir(cityOutputFolder);
        end

        % Check OSM file exists before loading viewer
        cityFolder = sprintf('%s_ROI', cityName);
        osmFile    = fullfile(osmFolder, cityFolder, ...
                        sprintf('%s_roi_%d.osm', cityName, roiID));

        if ~isfile(osmFile)
            fprintf('Missing OSM file: %s\n', osmFile);
            fclose('all');
            clear tx rx rtpm ss gridSS dset info existingDsets;
            gpuDevice([]);
            pause(0.5);
            continue;
        end

        % Load buildings once — viewer stays open for all 50 TX
        Buildings = readgeotable(osmFile, 'Layer', 'buildings');
        viewer    = siteviewer(Buildings=Buildings, Basemap='openstreetmap');

        % Track consecutive non-material GPU errors within this ROI
        gpuErrorCount = 0;

        for txIdx = 1:numTransmitters
            txlat    = tx_data.txLat(txIdx);
            txlon    = tx_data.txLon(txIdx);
            txHeight = tx_data.txHeight(txIdx);

            tx = txsite('Latitude', txlat, 'Longitude', txlon, 'AntennaHeight', txHeight);
            rx = rxsite('Latitude', lats,  'Longitude', lons,  'AntennaHeight', txHeight);

            try
                % --- Attempt 1: auto material, GPU on ---
                rtpm = propagationModel("raytracing", Method="sbr", ...
                    MaxNumReflections=R, BuildingsMaterial="auto");
                rtpm.UseGPU = "on";

                tic;
                ss = sigstrength(rx, tx, rtpm);
                elapsedTime = toc;

                gpuErrorCount = 0;  % reset on success

            catch ME_auto
                if contains(ME_auto.message, 'material')
                    % --- Attempt 2: material error → retry with concrete, GPU on ---
                    fprintf('🔄 Material fallback (concrete): ROI %d Tx %d\n', roiID, txIdx);
                    try
                        rtpm = propagationModel("raytracing", Method="sbr", ...
                            MaxNumReflections=R, BuildingsMaterial="concrete");
                        rtpm.UseGPU = "on";

                        tic;
                        ss = sigstrength(rx, tx, rtpm);
                        elapsedTime = toc;

                        gpuErrorCount = 0;  % material fallback succeeded

                    catch ME_concrete
                        % concrete also failed — log and skip this TX
                        gpuErrorCount = gpuErrorCount + 1;
                        fprintf('⚠️ Concrete fallback also failed ROI %d Tx %d: %s\n', ...
                            roiID, txIdx, ME_concrete.message);
                        failLog = fullfile(outputFolder, 'GPU_Failures_Log.txt');
                        fid = fopen(failLog, 'a');
                        fprintf(fid, '[%s] ROI %d (%s), Tx %d, R1B1 (concrete failed): %s\n', ...
                            datetime("now"), roiID, cityName, txIdx, ME_concrete.message);
                        fclose(fid);
                        continue;  % skip to next TX
                    end

                else
                    % --- Non-material error (OptiX / GPU crash) ---
                    gpuErrorCount = gpuErrorCount + 1;
                    fprintf('⚠️ Error R1B1 ROI %d (%s) Tx %d: %s\n', ...
                        roiID, cityName, txIdx, ME_auto.message);
                    failLog = fullfile(outputFolder, 'GPU_Failures_Log.txt');
                    fid = fopen(failLog, 'a');
                    fprintf(fid, '[%s] ROI %d (%s), Tx %d, R1B1: %s\n', ...
                        datetime("now"), roiID, cityName, txIdx, ME_auto.message);
                    fclose(fid);

                    % 3-strike rule: attempt GPU reset + CPU fallback
                    if gpuErrorCount >= 3
                        fprintf('🔁 %d consecutive GPU errors — resetting GPU for ROI %d...\n', ...
                            gpuErrorCount, roiID);
                        try close(viewer); catch, end
                        fclose('all');
                        gpuDevice([]);
                        pause(2);

                        try
                            Buildings2 = readgeotable(osmFile, 'Layer', 'buildings');
                            viewer     = siteviewer(Buildings=Buildings2, Basemap='openstreetmap');

                            % Try auto on CPU first, concrete on CPU as last resort
                            try
                                rtpm_cpu = propagationModel("raytracing", Method="sbr", ...
                                    MaxNumReflections=R, BuildingsMaterial="auto");
                                rtpm_cpu.UseGPU = "off";
                                tic;
                                ss = sigstrength(rx, tx, rtpm_cpu);
                                elapsedTime = toc;
                            catch
                                rtpm_cpu = propagationModel("raytracing", Method="sbr", ...
                                    MaxNumReflections=R, BuildingsMaterial="concrete");
                                rtpm_cpu.UseGPU = "off";
                                tic;
                                ss = sigstrength(rx, tx, rtpm_cpu);
                                elapsedTime = toc;
                            end

                            fprintf('✅ CPU fallback saved: ROI %d Tx %d\n', roiID, txIdx);
                            gpuErrorCount = 0;

                        catch ME_fallback
                            fprintf('❌ CPU fallback failed ROI %d Tx %d: %s\n', ...
                                roiID, txIdx, ME_fallback.message);
                            fid = fopen(failLog, 'a');
                            fprintf(fid, '[%s] CPU FALLBACK FAILED ROI %d (%s), Tx %d: %s\n', ...
                                datetime("now"), roiID, cityName, txIdx, ME_fallback.message);
                            fclose(fid);
                            continue;  % skip this TX
                        end
                    else
                        continue;  % skip this TX, try next
                    end
                end
            end

            % --- Write result (reached only if ss was computed successfully) ---
            ss(~isfinite(ss)) = -120;
            gridSS = flipud(reshape(ss, size(Xs)));

            dset = sprintf('/Tx_%d', txIdx);
            fileExists = isfile(h5_R1B1);
            if fileExists
                info = h5info(h5_R1B1);
                existingDsets = {info.Datasets.Name};
            else
                existingDsets = {};
            end
            if ~fileExists || ~any(strcmp(existingDsets, dset))
                h5create(h5_R1B1, dset, size(gridSS));
            end
            h5write(h5_R1B1, dset, gridSS);

            fid = fopen(logFile, 'a');
            fprintf(fid, '%s,%d,%d,R1B1,%.4f\n', cityName, roiID, txIdx, elapsedTime);
            fclose(fid);

            fprintf('Saved: ROI %d | Tx %d | R1B1\n', roiID, txIdx);
        end

        close(viewer);

        % Cleanup after each ROI
        fclose('all');
        clear viewer Buildings tx rx rtpm ss gridSS dset info existingDsets;
        gpuDevice([]);
        pause(0.5);

        % Periodic deep cleanup every 25 ROIs
        if mod(i, 25) == 0
            fprintf('\n🔄 Deep cleanup at ROI index %d...\n', i);
            fclose('all');
            clear viewer Buildings tx rx rtpm ss gridSS dset h5_R1B1 info existingDsets;
            gpuDevice([]);
            pause(2);
        end

    catch ME
        fprintf('‼️ Error processing ROI %d (%s): %s\n', roiID, cityName, ME.message);
        errorLog = fullfile(outputFolder, 'General_Error_Log.txt');
        fid = fopen(errorLog, 'a');
        fprintf(fid, '[%s] ROI %d (%s): %s\n', ...
            datetime("now"), roiID, cityName, getReport(ME, 'extended', 'hyperlinks', 'off'));
        fclose(fid);
        try reset(gpuDevice); catch, end
        continue;
    end

end

fprintf('\n=== ALL df ROIs COMPLETE ===\n');


