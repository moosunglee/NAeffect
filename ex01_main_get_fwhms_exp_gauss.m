%{
%% Example 01. 3D PSF Simulation
%% Press F5 to start the simulation, which performs
%% NA-dependent PSF generation & FWHM calculations.
%% Moosung Lee, University of Stuttgart, 2024.11.21
%}
%% 1. Set the directory and initialize
clc; clear; close all;

% Get the current directory of the active script
current_dirname = fileparts(matlab.desktop.editor.getActiveFilename);
main_dirname = current_dirname; 
addpath(genpath(fullfile(main_dirname))); % Add main directory and subfolders to the path

disp(['Main directory: ', main_dirname]); % Display the main directory

%% 2. Initialize major parameters
use_GPU = true;                      % Use GPU for computations if available
wavelength = 1.064;                  % Wavelength of light in micrometers
RI_bg = 1;                           % Refractive index of the background
NAs_theory = 0.1:0.01:0.98;          % Numerical apertures to evaluate
pitches = wavelength ./ linspace(1, 16, length(NAs_theory)); % Pitch values corresponding to NAs

% Preallocate result arrays for Full Width at Half Maximum (FWHM)
Dxs = []; Dzs = []; 
Dzs_dxx = []; Dxs_dxx = [];
powers = [];

% Gaussian function for fitting intensity profiles
gauss_fun = @(a, b, c, x) a * exp(-(x - b).^2 ./ (2 * c^2)); 
w0_NA = 0.7835; % Beam waist parameter for Gaussian beam

%% 3. Loop through each NA and compute results
for j1 = 1:length(NAs_theory)
    %% Initialize parameters for the current NA
    NA = NAs_theory(j1);             % Current numerical aperture
    pitch = pitches(j1);             % Pitch corresponding to this NA
    FOV = [101, 101, 201] * pitch;   % Field of View (FOV) in micrometers

    % Initialize the solver for the computation
    solver = cbs.init_solver('use_GPU', use_GPU, 'NA', 1, 'wavelength0', wavelength, ...
                             'RI_bg', RI_bg, 'pitch', pitch, 'FOV', FOV, 'dimension', 3);

    %% Compute the analytic field using Novotny's formulation
    E_analytic = cbs.field.gaussian_analytic_v2(solver, ...
                 'polarization', [1, 1i], 'NA', NA, 'w0_NA', w0_NA);
    I_analytic = sum(abs(E_analytic).^2, 4); % Compute intensity from field
    I_analytic = I_analytic ./ max(I_analytic(:)); % Normalize intensity

    %% Analyze the intensity profile
    [fwhm_x, fwhm_z] = process_lines(I_analytic, gauss_fun, pitch, NA);

    % Store results
    Dxs(j1) = fwhm_x; 
    Dzs(j1) = fwhm_z; 
end

%% 4. Plot results
close all;
figure;

% Plot Dxs (FWHM along x-axis) vs NAs
subplot(1, 2, 1);
plot(NAs_theory, Dxs, 'rx', 'DisplayName', 'FWHM_x (computed)');
hold on;
plot(NAs_theory, wavelength * 0.51 ./ NAs_theory, 'k', 'DisplayName', 'Theoretical (0.51λ/NA)');
xlabel('Numerical Aperture (NA)');
ylabel('FWHM_x (μm)');
legend('show');
title('FWHM_x vs NA');
grid on;

% Plot Dzs (FWHM along z-axis) vs NAs
subplot(1, 2, 2);
plot(NAs_theory, Dzs, 'rx', 'DisplayName', 'FWHM_z (computed)');
hold on;
plot(NAs_theory, 0.88 * wavelength ./ (1 - sqrt(1 - NAs_theory.^2)), 'k', 'DisplayName', 'Theoretical (0.88λ / (1 - √(1 - NA²)))');
plot(NAs_theory, 4 / pi * wavelength ./ NAs_theory.^2, 'k:', 'DisplayName', 'Alternative (4λ / (πNA²))');
xlabel('Numerical Aperture (NA)');
ylabel('FWHM_z (μm)');
legend('show');
title('FWHM_z vs NA');
xlim([0.5 1]);
grid on;

%% 5. Save results
save(fullfile(main_dirname, 'subcodes', 'analytic_fwhm_apo.mat'), 'NAs_theory', 'Dxs', 'Dzs', 'Dzs_dxx', 'Dxs_dxx', 'powers');

disp('Results saved to analytic_fwhm_apo.mat');

%% Subfunctions
function [fwhm_x, fwhm_z] = process_lines(I_analytic, gauss_fun, pitch, NA)
    % Extract lines
    line_z = extract_line(I_analytic, 3);
    line_x = extract_line(I_analytic, 1);
    
    % Process lines
    line_x = process_line(line_x, 1/3);
    line_z = process_line(line_z, 1/3);

    % Fit and calculate FWHM for X and Z
    fwhm_x = fit_and_plot(line_x, gauss_fun, pitch, 221); title('X lineplot');
    fwhm_z = fit_and_plot(line_z, gauss_fun, pitch, 222); title('Z lineplot');
    fprintf('NA: %.2f, FWHM_x: %.4f, FWHM_z: %.4f\n', NA, fwhm_x, fwhm_z);

    % Define the middle indices for slicing
    mid_x = floor(size(I_analytic, 1) / 2) + 1; % Middle index along X
    mid_y = floor(size(I_analytic, 2) / 2) + 1; % Middle index along Y
    mid_z = floor(size(I_analytic, 3) / 2) + 1; % Middle index along Z
    
    % Extract XY and XZ cross-sections
    xy_slice = squeeze(I_analytic(:, :, mid_z)); % XY plane at mid-Z
    xz_slice = squeeze(I_analytic(:, mid_y, :)); % XZ plane at mid-Y


    % Display XY cross-section
    subplot(223);
    imagesc(xy_slice);
    axis image; % Ensure correct aspect ratio
    colorbar;  % Add color bar for intensity scale
    title('XY Cross-Section');
    xlabel('X');
    ylabel('Y');
    
    % Display XZ cross-section
    subplot(224);
    imagesc(xz_slice');
    axis image; % Ensure correct aspect ratio
    colorbar;  % Add color bar for intensity scale
    title('XZ Cross-Section');
    xlabel('X');
    ylabel('Z');
    % sgtitle(fprintf('NA: %.2f, FWHM_x: %.4f, FWHM_z: %.4f\n', NA, fwhm_x, fwhm_z))
    colormap hot
    set(gcf, 'color', 'w')
    drawnow

end

function line = extract_line(I_analytic, dim)
    % Extract a line from I_analytic along the specified dimension
    mid = floor(size(I_analytic) / 2) + 1;
    if dim == 1
        line = squeeze(I_analytic(:, mid(2), mid(3)));
    elseif dim == 3
        line = squeeze(I_analytic(mid(1), mid(2), :));
    else
        error('Unsupported dimension for line extraction.');
    end
end

function processed_line = process_line(line, target_value)
    % Find indices around the target value and truncate the line
    idx = find_closest_indices(line, target_value);
    processed_line = line(idx(1):idx(end));
end

function idx = find_closest_indices(line, target_value)
    % Find indices closest to the target value
    [~, idx(1)] = min(abs(line - target_value));
    idx(2) = round(length(line) / 2 + abs(length(line) / 2 - idx(1)));
    idx(3) = round(length(line) / 2 - abs(length(line) / 2 - idx(1)));
    idx = sort(idx);
    idx = min(max(idx, 1), length(line));
end

function fwhm = fit_and_plot(line, gauss_fun, pitch, subplot_position)
    % Fit a Gaussian to the line and calculate FWHM
    x0 = (1:length(line)) - mean(1:length(line));
    fit_result = fit(x0', line, gauss_fun, ...
        'StartPoint', [max(line), mean(x0), std(x0)]);
    fwhm = 2 * sqrt(2 * log(2)) * fit_result.c * pitch;

    % Plot the result
    subplot(subplot_position);
    plot(fit_result, x0, line);
end