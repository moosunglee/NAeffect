%{
%% Example 02. Figure 3b & 3c fitting figures
%% Press F5 to start the simulation, which performs
%% predictions of Ωz/Ωx ratios (3b) & Laser Power (3c)
%% Moosung Lee, University of Stuttgart, 2024.11.21
%}
%% 01. Directory and Initialization
% Clear workspace and close figures
clc; clear; close all;

% Set current directory to script location and add subdirectories
cd0 = fileparts(matlab.desktop.editor.getActiveFilename);
cd(cd0);
addpath(genpath(fullfile(cd0, 'subcodes')));

%% 02. Constants and Parameters
% Load experimental data (optional)
% load(fullfile(cd0, 'data', 'fig02_resolution'))

% Define physical constants
epsilon0 = 8.8541878128e-12; % [F/m] Vacuum permittivity
c = 299792458;              % [m/s] Speed of light

% Particle properties
RI_sp = 1.4496;             % Refractive index of the particle
radius = 142 / 2 * 1e-9;    % Particle radius [m]
rho = 1850;                 % Particle density [kg/m^3]
alpha = 3 * (4 * pi / 3 * radius^3) * epsilon0 * (RI_sp^2 - 1) / (RI_sp^2 + 2); 
mass = (4 * pi / 3 * radius^3) * rho; % Particle mass [kg]

% Optical parameters
wavelength = 1064e-9;       % Laser wavelength [m]
w0_NA = 0.7835;             % Waist parameter
NAs_theory = 0.01:0.01:1;   % Range of numerical apertures (NA)

% Load simulation and analytic results
load(fullfile(cd0, 'subcodes', 'fig03_exp_data.mat')); % Experimental data
data_analytic = load(fullfile(cd0, 'subcodes', 'analytic_fwhm_apo.mat')); % Analytic FWHM data

%% 03. Theoretical Calculations
% Calculate Gaussian beam parameters
beta = sqrt(2);             % Beam parameter
wx = beta * wavelength / pi ./ NAs_theory; % Beam waist along x [m]
z0 = pi * wx.^2 / wavelength; % Rayleigh range [m]

% Theoretical frequencies (x and z directions)
freqs_X_kHz_theory = sqrt(12 * pi^3 * (RI_sp^2 - 1) ./ (c * rho * (RI_sp^2 + 2))) .* ...
                     NAs_theory.^2 ./ beta^2 / wavelength^2;
freqs_Z_kHz_theory = sqrt(6 * pi^3 * (RI_sp^2 - 1) ./ (c * rho * (RI_sp^2 + 2))) .* ...
                     NAs_theory.^3 ./ beta^3 / wavelength^2;

% Alternate theoretical frequencies with fixed scaling
freqs_X_kHz_theory0 = freqs_X_kHz_theory * beta^2; 
freqs_Z_kHz_theory0 = freqs_Z_kHz_theory * beta^3;

% Extract analytic data for comparison
wxs_analytic = 1e-6 * data_analytic.Dxs / sqrt(2 * log(2)); % Convert to meters
wzs_analytic = 1e-6 * data_analytic.Dzs / sqrt(2 * log(2)); 
freqs_X_kHz_analytic = sqrt(12 * (RI_sp^2 - 1) ./ (pi * c * rho * (RI_sp^2 + 2))) ./ wxs_analytic.^2;
freqs_Z_kHz_analytic = sqrt(12 * (RI_sp^2 - 1) ./ (pi * c * rho * (RI_sp^2 + 2))) ./ ...
                        (wxs_analytic .* wzs_analytic);

%% 04. Plot Figure 3b: Frequency Ratio (fz/fx)
figure;
hold on;

% Plot experimental data for different particles
points = ['^sv']; % Marker styles
for j1 = 1:3
    plot(NAs_exp(labels_particle == j1), ...
         freqs_Z_kHz_exp(labels_particle == j1) ./ freqs_X_kHz_exp(labels_particle == j1), ...
         ['b', points(j1)], 'MarkerSize', 8);
end

% Plot theoretical predictions
plot(NAs_theory, freqs_Z_kHz_theory ./ freqs_X_kHz_theory, 'k', 'DisplayName', 'Theory');
plot(NAs_theory, freqs_Z_kHz_theory0 ./ freqs_X_kHz_theory0, 'k:', 'DisplayName', 'Theory (alt)');
plot(data_analytic.NAs_theory, freqs_Z_kHz_analytic ./ freqs_X_kHz_analytic, 'g', 'DisplayName', 'Analytic');

% Customize plot
xlim([0.3, 0.95]);
ylim([0.1, 0.5]);
xlabel('NA');
ylabel('f_z / f_x ratio');
legend('show');
set(gcf, 'Color', 'w');

%% 05. Compute and Plot Power Ratios
% Constants for silica particles
rho_sil = 1850; % Silica density [kg/m^3]
RI_sil = 1.4496; % Silica refractive index

% Preallocate results
Ps_Gauss_x = []; Ps_Gauss_z = [];
Ps_Simul_x = []; Ps_Simul_z = [];
NAs_plot = []; points_plot = [];

for j1 = 1:3
    % Extract analytic widths for corresponding experimental NAs
    wxs = 1e-6 * data_analytic.Dxs(round(NAs_exp(labels_particle == j1) * 100) - 9) / sqrt(2 * log(2));
    wzs = 1e-6 * data_analytic.Dzs(round(NAs_exp(labels_particle == j1) * 100) - 9) / sqrt(2 * log(2));
    Ps = (1 - exp(-NAs_exp(labels_particle == j1).^2 / w0_NA^2)) * 1.1376^2;

    % Experimental frequencies
    Omegas_z = freqs_Z_kHz_exp(labels_particle == j1) * 2 * pi * 1000;
    Omegas_x = freqs_X_kHz_exp(labels_particle == j1) * 2 * pi * 1000;
    NAs = NAs_exp(labels_particle == j1);

    % Store results
    NAs_plot = [NAs_plot, NAs];
    points_plot = [points_plot, repmat(points(j1), 1, length(NAs))];

    % Calculate Gaussian-based powers
    beta = sqrt(2);
    Ps_Gauss_x = [Ps_Gauss_x, rho * c / (12 * pi^3) * (RI_sil^2 + 2) / (RI_sil^2 - 1) .* ...
                  beta^4 * wavelength^4 .* Omegas_x.^2 ./ NAs.^4];
    Ps_Gauss_z = [Ps_Gauss_z, rho * c / (6 * pi^3) * (RI_sil^2 + 2) / (RI_sil^2 - 1) .* ...
                  beta^6 * wavelength^4 .* Omegas_z.^2 ./ NAs.^6];

    % Simulation-based powers
    Ps_Simul_x = [Ps_Simul_x, pi * c * rho / 12 * (RI_sil^2 + 2) / (RI_sil^2 - 1) .* ...
                  (wxs.^2 .* Omegas_x).^2];
    Ps_Simul_z = [Ps_Simul_z, pi * c * rho / 12 * (RI_sil^2 + 2) / (RI_sil^2 - 1) .* ...
                  (wxs .* wzs .* Omegas_z).^2];
end

% Fit power data using Gaussian model
gauss_fun = @(a, x) a .* (1 - exp(-x.^2 / w0_NA^2));
fit1_x = fit(NAs_plot', Ps_Simul_x', gauss_fun, 'StartPoint', 1000);
fit1_z = fit(NAs_plot', Ps_Simul_z', gauss_fun, 'StartPoint', 1000);

% Plot results
figure('Renderer', 'painters', 'Position', [10, 10, 977, 847]);
for j1 = 1:length(points_plot)
    % plot(NAs_plot(j1), Ps_Gauss_x(j1)*1000, points_plot(j1), 'MarkerFaceColor',[0 0 1],...
    %     'MarkerEdgeColor',[0 1 1],'MarkerSize',8), hold on, 
    % plot(NAs_plot(j1), Ps_Gauss_z(j1)*1000, points_plot(j1), 'MarkerFaceColor',[1 0 0],...
    %     'MarkerEdgeColor',[1 0 1],'MarkerSize',8), hold on, 
    plot(NAs_plot(j1), Ps_Simul_x(j1)*1000, points_plot(j1), 'MarkerFaceColor',[0 1 1],...
        'MarkerEdgeColor',[0 0 1],'MarkerSize',8), hold on, 
    % plot(NAs_plot(j1), Ps_Simul_z(j1)*1000, points_plot(j1), 'MarkerFaceColor',[1 0 1],...
    %     'MarkerEdgeColor',[1 0 0],'MarkerSize',8), hold on, 
end
% plot(NAs_theory, (1-exp(-NAs_theory.^2/w0_NA^2))*1.171*1000, 'c') % Simul
% plot(NAs_theory, (1-exp(-NAs_theory.^2/w0_NA^2))*0.58*1000, 'c') % Simul
plot(NAs_theory, (1-exp(-NAs_theory.^2/w0_NA^2))*1.291*1000, 'k') % Simul
% plot(NAs_theory, (1-exp(-NAs_theory.^2/w0_NA^2))*1.3*1000, 'k') % Simul

plot(NAs_theory, (1 - exp(-NAs_theory.^2 / w0_NA^2)) * 1.291 * 1000, 'k');
xlim([0.3, 0.95]);
ylim([0, 1000]);
