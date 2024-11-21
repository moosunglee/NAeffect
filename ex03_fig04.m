%{
%% Example 02. Figure 4 fitting figures
%% Press F5 to start the simulation, which performs
%% predictions of Ωz, Ωx (4a), scattering power (4b), and recoil heating rate (4c)
%% Moosung Lee, University of Stuttgart, 2024.11.21
%}
%% 01. Set Directory and Initialize
clc; clear; close all;

% Set working directory to the location of this script
cd0 = fileparts(matlab.desktop.editor.getActiveFilename);
cd(cd0);

% Add all subdirectories in 'Codes' to the MATLAB path
addpath(genpath(fullfile(cd0, 'subcodes')));

%% Load Data and Set Parameters
% Load analytic and experimental data
data_analytic = load(fullfile(cd0, 'subcodes', 'analytic_fwhm_apo.mat'));
load(fullfile(cd0, 'subcodes', 'fig03_exp_data.mat'));


% Numerical Aperture (NA) values from theory
NAs = data_analytic.NAs_theory;
NAs_theory = NAs;

% Fundamental constants and parameters
c = 299792458;                % Speed of light [m/s]
bead_radius = 71e-9;          % Bead radius [m]
wavelength = 1064e-9;         % Laser wavelength [m]
w0_NA = 0.7835;               % Normalization factor for NA
Power = 0.25;                 % Laser power [W]
Volume = (4/3) * pi * bead_radius^3; % Bead volume [m^3]
rho = 1850;                   % Density of the bead [kg/m^3]
mass = rho * Volume;          % Mass of the bead [kg]
k = 2 * pi / wavelength;      % Wavenumber [m^-1]
w0 = c / wavelength * 2 * pi; % Angular frequency of light [rad/s]
RI = 1.4496;                  % Refractive index of the bead
RI_sp = RI;                   % Shortcut for refractive index

%% Figure 4a: Frequency Comparison
figure('Renderer', 'painters', 'Position', [10, 10, 750, 645]);

% Theoretical trap frequencies for X and Z directions
beta = sqrt(2); % Beam shape factor
freqs_X_kHz_theory = sqrt(12 * pi^3 * (RI_sp^2 - 1) / (c * rho * (RI_sp^2 + 2))) .* NAs_theory.^2 / beta^2 / wavelength^2;
freqs_Z_kHz_theory = sqrt(6 * pi^3 * (RI_sp^2 - 1) / (c * rho * (RI_sp^2 + 2))) .* NAs_theory.^3 / beta^3 / wavelength^2;

% Paraxial limit (beta = 1)
freqs_X_kHz_theory0 = sqrt(12 * pi^3 * (RI_sp^2 - 1) / (c * rho * (RI_sp^2 + 2))) .* NAs_theory.^2 / wavelength^2;
freqs_Z_kHz_theory0 = sqrt(6 * pi^3 * (RI_sp^2 - 1) / (c * rho * (RI_sp^2 + 2))) .* NAs_theory.^3 / wavelength^2;

% Analytic trap frequencies
wxs_analytic = 1e-6 * data_analytic.Dxs / sqrt(2 * log(2)); % Beam waist (x) [m]
wzs_analytic = 1e-6 * data_analytic.Dzs / sqrt(2 * log(2)); % Beam waist (z) [m]
freqs_X_kHz_analytic = sqrt(12 * (RI_sp^2 - 1) / (pi * c * rho * (RI_sp^2 + 2))) ./ wxs_analytic.^2;
freqs_Z_kHz_analytic = sqrt(12 * (RI_sp^2 - 1) / (pi * c * rho * (RI_sp^2 + 2))) ./ wxs_analytic ./ wzs_analytic;

% Plot theoretical and analytic frequencies
plot(NAs_theory, freqs_Z_kHz_theory / 1000 / (2 * pi) * sqrt(Power), 'r', 'DisplayName', 'Theory (Z)'), hold on;
plot(NAs_theory, freqs_X_kHz_theory / 1000 / (2 * pi) * sqrt(Power), 'b', 'DisplayName', 'Theory (X)');
plot(NAs_theory, freqs_Z_kHz_theory0 / 1000 / (2 * pi) * sqrt(Power), 'r:', 'DisplayName', 'Paraxial (Z)');
plot(NAs_theory, freqs_X_kHz_theory0 / 1000 / (2 * pi) * sqrt(Power), 'b:', 'DisplayName', 'Paraxial (X)');
plot(data_analytic.NAs_theory, freqs_X_kHz_analytic / 1000 / (2 * pi) * sqrt(Power), 'c', 'LineWidth', 3, 'DisplayName', 'Analytic (X)');
plot(data_analytic.NAs_theory, freqs_Z_kHz_analytic / 1000 / (2 * pi) * sqrt(Power), 'm', 'LineWidth', 3, 'DisplayName', 'Analytic (Z)');

% Formatting
xlim([0.3, 0.95]);
ylim([0, 600]);
xlabel('NA');
ylabel('Frequency (kHz)');
legend('show');
grid on;

%% Figure 4b: Scattering Power
figure('Renderer', 'painters', 'Position', [10, 10, 750, 645]);

% Rayleigh scattering power for different beta values
for beta = [sqrt(2), 1]
    wxs = wavelength * beta / pi ./ data_analytic.NAs_theory;
    P_scat_tot = 3 * Volume^2 * k^4 * Power ./ (pi^2 * wxs.^2) .* (RI^2 - 1)^2 / (RI^2 + 2)^2;
    plot(NAs, P_scat_tot * 1e6, 'DisplayName', sprintf('Beta = %.1f', beta)); hold on;
end

% Analytic scattering power
wxs_analytic = data_analytic.Dxs / (sqrt(2 * log(2))) * 1e-6;
P_scat_tot = 3 * Volume^2 * k^4 * Power ./ (pi^2 * wxs_analytic.^2) .* (RI^2 - 1)^2 / (RI^2 + 2)^2;
plot(NAs, P_scat_tot * 1e6, 'b', 'DisplayName', 'Analytic');

% Formatting
xlim([0.3, 0.95]);
ylim([0, 60]);
xlabel('NA');
ylabel('Scattering Power (µW)');
legend('show');
grid on;

%% Figure 4c: Damping Rates (Gamma)
figure('Renderer', 'painters', 'Position', [10, 10, 750, 645]);

% Damping rates GammaX and GammaZ for different beta values
for beta = [sqrt(2), 1]
    wxs = wavelength * beta / pi ./ data_analytic.NAs_theory;
    Omega_X = sqrt(12 * pi^3 * (RI_sp^2 - 1) ./ (c * rho * (RI_sp^2 + 2)) .* Power) .* NAs_theory.^2 / beta^2 / wavelength^2;
    Omega_Z = sqrt(6 * pi^3 * (RI_sp^2 - 1) ./ (c * rho * (RI_sp^2 + 2)) .* Power) .* NAs_theory.^3 / beta^3 / wavelength^2;
    P_scat_tot = 3 * Volume^2 * k^4 * Power ./ (pi^2 * wxs.^2) .* (RI^2 - 1)^2 / (RI^2 + 2)^2;
    GammaX = P_scat_tot ./ (5 * mass * c^2) * w0 ./ Omega_X;
    GammaZ = P_scat_tot ./ (5 * mass * c^2) * w0 ./ Omega_Z;
    plot(NAs, GammaX / (2 * pi * 1000), 'r', 'DisplayName', sprintf('GammaX (Beta = %.1f)', beta)); hold on;
    plot(NAs, GammaZ / (2 * pi * 1000), 'b', 'DisplayName', sprintf('GammaZ (Beta = %.1f)', beta));
end

% Formatting
xlim([0.3, 0.95]);
xlabel('NA');
ylabel('Damping Rate (kHz)');
legend('show');
grid on;