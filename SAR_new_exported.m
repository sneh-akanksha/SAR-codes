classdef SAR_new_exported < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure                        matlab.ui.Figure
        TabGroup                        matlab.ui.container.TabGroup
        ParametersTab                   matlab.ui.container.Tab
        RotationalErrorCheckBox         matlab.ui.control.CheckBox
        TranslationalErrorCheckBox      matlab.ui.control.CheckBox
        FineMotionCompensationDropDown  matlab.ui.control.DropDown
        FineMotionCompensationDropDownLabel  matlab.ui.control.Label
        CoarseMotionCompensationDropDown  matlab.ui.control.DropDown
        CoarseMotionCompensationDropDownLabel  matlab.ui.control.Label
        RootMeanSquareErrorCrossrangemEditField  matlab.ui.control.NumericEditField
        RootMeanSquareErrorCrossrangemEditFieldLabel  matlab.ui.control.Label
        ConsoleTextArea                 matlab.ui.control.TextArea
        ConsoleTextAreaLabel            matlab.ui.control.Label
        NoiseTypeDropDown               matlab.ui.control.DropDown
        NoiseTypeDropDownLabel          matlab.ui.control.Label
        ResultsandMetricLabel           matlab.ui.control.Label
        RootMeanSquareErrorRangemEditField  matlab.ui.control.NumericEditField
        RootMeanSquareErrorRangemEditFieldLabel  matlab.ui.control.Label
        PeakSideLobeRatiodBEditField    matlab.ui.control.NumericEditField
        PeakSideLobeRatiodBEditFieldLabel  matlab.ui.control.Label
        IntegratedSideLobeRatioEditField  matlab.ui.control.NumericEditField
        IntegratedSideLobeRatioEditFieldLabel  matlab.ui.control.Label
        EntropyEditField                matlab.ui.control.NumericEditField
        EntropyEditFieldLabel           matlab.ui.control.Label
        ContrastEditField               matlab.ui.control.NumericEditField
        ContrastEditFieldLabel          matlab.ui.control.Label
        Image                           matlab.ui.control.Image
        RunButton                       matlab.ui.control.Button
        RadarandRadarPlatformParametersLabel  matlab.ui.control.Label
        HalfSwathWidthalongtrackmEditField  matlab.ui.control.NumericEditField
        HalfSwathWidthalongtrackmEditFieldLabel  matlab.ui.control.Label
        DistancetocenterofswarthmEditField  matlab.ui.control.NumericEditField
        DistancetocenterofswarthmEditFieldLabel  matlab.ui.control.Label
        CoherentProcessingIntervalsEditField  matlab.ui.control.NumericEditField
        CoherentProcessingIntervalsEditFieldLabel  matlab.ui.control.Label
        PulseRepitionFrequencyHzEditField  matlab.ui.control.NumericEditField
        PulseRepitionFrequencyHzEditFieldLabel  matlab.ui.control.Label
        AltitudemEditField              matlab.ui.control.NumericEditField
        AltitudemEditFieldLabel         matlab.ui.control.Label
        PlatformvelocitymsEditField     matlab.ui.control.NumericEditField
        PlatformvelocitymsEditFieldLabel  matlab.ui.control.Label
        CarrierFrequencyGHzEditField    matlab.ui.control.NumericEditField
        CarrierFrequencyGHzEditFieldLabel  matlab.ui.control.Label
        BandwidthMHzEditField           matlab.ui.control.NumericEditField
        BandwidthMHzEditFieldLabel      matlab.ui.control.Label
        SquintAngledegreesEditField     matlab.ui.control.NumericEditField
        SquintAngledegreesEditFieldLabel  matlab.ui.control.Label
        SignaltoNoiseRatiodBEditField   matlab.ui.control.NumericEditField
        SignaltoNoiseRatiodBEditFieldLabel  matlab.ui.control.Label
        DutyCycleEditField              matlab.ui.control.NumericEditField
        DutyCycleEditFieldLabel         matlab.ui.control.Label
        TranslationalandRotationalMotionErrorsLabel  matlab.ui.control.Label
        CrosstrackerrormEditField       matlab.ui.control.NumericEditField
        CrosstrackerrormEditFieldLabel  matlab.ui.control.Label
        AzimuthalongtrackerrormEditField  matlab.ui.control.NumericEditField
        AzimuthalongtrackerrormEditFieldLabel  matlab.ui.control.Label
        ElevationerrormEditField        matlab.ui.control.NumericEditField
        ElevationerrormEditFieldLabel   matlab.ui.control.Label
        RolldegreesEditField            matlab.ui.control.NumericEditField
        RolldegreesEditFieldLabel       matlab.ui.control.Label
        PitchdegreesEditField           matlab.ui.control.NumericEditField
        PitchdegreesEditFieldLabel      matlab.ui.control.Label
        YawdegreesEditField             matlab.ui.control.NumericEditField
        YawdegreesEditFieldLabel        matlab.ui.control.Label
        TargetParametersLabel           matlab.ui.control.Label
        CrosstrackmEditField            matlab.ui.control.NumericEditField
        CrosstrackmEditFieldLabel       matlab.ui.control.Label
        AzimuthalongtrackmEditField     matlab.ui.control.NumericEditField
        AzimuthalongtrackmEditFieldLabel  matlab.ui.control.Label
        ElevationmEditField             matlab.ui.control.NumericEditField
        ElevationmEditFieldLabel        matlab.ui.control.Label
        RadarCrossSectiondBsmEditField  matlab.ui.control.NumericEditField
        RadarCrossSectiondBsmEditFieldLabel  matlab.ui.control.Label
        SARMotionCompensationAnalysisforPointTargetsLabel  matlab.ui.control.Label
        UIAxes                          matlab.ui.control.UIAxes
        ReferenceTab                    matlab.ui.container.Tab
        Image3                          matlab.ui.control.Image
        Image2                          matlab.ui.control.Image
    end

methods (Access = private)
        
        function runSAR(app)

     
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % PLATFORM PARAMETERS
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Low squint angle (degrees)            
            theta_sq = app.SquintAngledegreesEditField.Value;
            % Altitude (m)
            z_0 = app.AltitudemEditField.Value;
            % Range distance to center of swath area (m)
            xc = app.DistancetocenterofswarthmEditField.Value; 
            %Mean Velocity of platform (along track - component)(m/s)
            vp = app.PlatformvelocitymsEditField.Value; 
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Errors
            
            % Translational motion errors
            if app.TranslationalErrorCheckBox.Value
                trans_err_flag = 1;
            else
                trans_err_flag = 0;
            end
            %Standard Deviation of Velocity of platform (along track - component) with error(m/s)
            std_vp = app.AzimuthalongtrackerrormEditField.Value;
            %Mean Velocity of platform (cross track azimuth - component)(m/s)
            vx = 0;
            %Standard Deviation of Velocity of platform (cross track azimuth - component) with error(m/s)
            std_vx = app.CrosstrackerrormEditField.Value;
            %Mean Velocity of platform (cross track elevation - component)(m/s)
            vz = 0;
            %Standard Velocity of platform (cross track elevation - component) with error(m/s)
            std_vz = app.ElevationerrormEditField.Value;
            
            % Rotational Motion Errors
            if app.RotationalErrorCheckBox.Value
                rot_err_flag = 1;
            else
                rot_err_flag = 0;
            end
            %Mean of Yaw angle (degrees)
            yaw = app.YawdegreesEditField.Value;
            % Standard Deviation of Yaw Angle
            std_yaw = 0;
            %Pitch angle (degrees)
            pitch = app.PitchdegreesEditField.Value;
            % Standard Deviation of Pitch Angle
            std_pitch = 0;
            %RCS (dbsm)
            RCS = app.RadarCrossSectiondBsmEditField.Value;
            %Noise Type
            selectedNoise = app.NoiseTypeDropDown.Value;
    
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % RADAR PARAMETERS
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %Carrier frequency (Hz)
            fc = 1e9*app.CarrierFrequencyGHzEditField.Value;
            %Baseband bandwidth (Hz)
            Bw = 1e6*app.BandwidthMHzEditField.Value;
            %Pulse Repetition Frequency (Hz)
            PRF = app.PulseRepitionFrequencyHzEditField.Value;
            %Total time (CPI) (sec)
            CPI = app.CoherentProcessingIntervalsEditField.Value;
            %Antenna length actual (m)
            La = 2*vp*(1/PRF);
            %Duty Cycle
            DC = app.DutyCycleEditField.Value;
            %Chirp Pulse Duration
            Tp = DC/PRF;
            % Signal to noise ratio (dB)
            SNR = app.SignaltoNoiseRatiodBEditField.Value;
            % Range envelope
            w_r = 1;
            % Azimuth envelope
            w_a = 1;
            % Half sawth width (Target is located within [Xc-X0,Xc+X0])
            R0 = app.HalfSwathWidthalongtrackmEditField.Value; 
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %TARGET PARAMETERS
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Number of targets
            ntarget= 1;
            %Range of target (m)
            xt = app.CrosstrackmEditField.Value;
            %Cross range (m)
            yt = app.AzimuthalongtrackmEditField.Value;
            % Height of target
            zt = app.ElevationmEditField.Value;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % MOTION COMPENSATION
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Coarse motion compensation
            
            % image type
            
            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % GLOBAL CONSTANTS
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Propagation speed
            c = 3e8;
            % Propagation frequency
            ic = 1 / c;
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % DERIVED VARIABLES
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Wavelength (m)
            lambda = c / fc; 
            % Range Chirp Rate (Hz/s)
            Kr = Bw / Tp; 
            % Slant range to center of swath area (m)
            Rc = sqrt(z_0^2 + xc^2);
            % Slant range (m)
            R_t = sqrt((z_0^2) + (xt^2));
            % Azimuth parameter Linear Azimuth FM rate
            Ka = (2 * vp^2) / (lambda * Rc);
            % Time Domain Sampling Interval (sec)
            ts = 1 / (2 * Bw);
            % Start time of sampling (sec)
            Ts = (2 * (Rc - R0)) / c; 
            % End time of sampling (sec)
            Tf = (2 * (Rc + R0)) / c;
            % Number of time (Range) samples)
            rbins = ceil(((Tf - Ts)) / ts); 
            t_fast = Ts + (0:rbins -1) * ts; 
            %Slow Time Array
            t_slow = linspace(0, CPI, PRF * CPI)'; 
            % Actual velocity of platform (while incorporating errors)
            vp_e = std_vp .* randn(length(t_slow), 1) + vp; 
            vx_e = std_vx .* randn(length(t_slow), 1) + vx; 
            vz_e = std_vz .* randn(length(t_slow), 1) + vz;
            % Rotational motion errors
            yaw_e = std_yaw .* ones(length(t_slow), 1) + yaw;
            pitch_e = std_pitch .* ones(2000, 1) + pitch;
    
            % Global constants
            y_0 = 0;
            x_0 = 0;
    
            % RCS in linear
            RCS_lin = exp(RCS/10);
            % Clear axes
            cla(app.UIAxes)

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % MODELING THE RECCEIVED SIGNAL BY RADAR
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            s_rx=zeros(PRF*CPI,rbins); 
            
            % LOS unit vector
            eta_los = [0; -sind(theta_sq); -cosd(theta_sq)];    
            % Coarse motion compensation error
            del_R_all = zeros(1,length(t_slow));
            
            % Trajectory followed by UAV with no error
            for j=1:length(t_slow)
            
                % Reinitialize the constants
                % Angle caused by rotational error with ideal
                beta = 0;
                % Direction vector caused by rotational error
                eta_yaw_pitch = [0 0 0];
                % Cross range shift caused by rotational error
                tgt_shift = 0;
            
                % Trajectory followed by UAV with no error   
                x_points(:, j) = x_0 + (vx * (CPI / 2 - t_slow(j)));
                y_points(:, j) = y_0 + (vp * (CPI / 2 - t_slow(j)));
                z_points(:, j) = z_0 + (vz * (CPI / 2 - t_slow(j)));
               
                % ideal range of target with no motion error (r_mo)
                range_ideal = sqrt(((zt - z_points(j))^2) + ((xt - x_points(j))^2) + ((yt - y_points(j))^2));
                range_wtme = range_ideal;
                tgt_shift = 0;
            
                if trans_err_flag
                % Trajectory followed by UAV with translational error 
                    x_points_e(:, j) = x_0 + (vx_e(j) * (CPI / 2 - t_slow(j)));
                    y_points_e(:, j) = y_0 + (vp_e(j) * (CPI / 2 - t_slow(j)));
                    z_points_e(:, j) = z_0 + (vz_e(j) * (CPI / 2 - t_slow(j)));        
            
                 range_wtme =  sqrt(((zt-z_points_e(j))^2)+((xt-x_points_e(j))^2)+((yt-y_points_e(j))^2));
                end
            
               if rot_err_flag
               % Trajectory followed by UAV with rotational error 
                    eta_yaw_pitch= [(deg2rad(yaw_e(j)) * sind(theta_sq)) - (deg2rad(pitch_e(j)) * cosd(theta_sq)); -sind(theta_sq) - (deg2rad(yaw_e(j)) * deg2rad(pitch_e(j)) * cosd(theta_sq)); -cosd(theta_sq)];       
                    % shifted squint angle (radians)
                    beta = abs(acos(dot(eta_los,eta_yaw_pitch)));
                    tgt_shift = tan(beta)*range_ideal;
               end
            
                % range of target with translationa and rotational motion error 
                range_wtrme = sqrt(range_wtme^2+tgt_shift^2);
                % Range error for coarse motion compensation
                del_R_all(j) = (range_wtrme-range_ideal )*cosd(beta);    
                td = t_fast-2*(range_wtrme)/c;
            
                % Final received signal for real trajectory                
               s_rx(j,:)=s_rx(j,:)+ RCS_lin.*w_r.*w_a.*exp(-1j*(4*pi*fc*ic*(range_wtrme))+1j*pi*Kr*(td.^2));                 
                  
            end
            
            % Add noise
            if strcmp(app.NoiseTypeDropDown.Value, 'White Gaussian Noise')
                         % Add awgn noise
                         s_rx_noise = awgn(s_rx,SNR);
            
                        elseif strcmp(app.NoiseTypeDropDown.Value, 'Speckle Noise')
                        % Add speckle noise
                         phi_noise = 2 * pi * rand(length(t_slow) ,length(t_fast));
                         var_speckle = 1;
                         s_rx_noise = s_rx+(var_speckle.*s_rx.*exp(1j.*phi_noise));
                        else
                            msg ="Error in noise type";
                            app.ConsoleTextArea.Value = msg;
            
                        end
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%           
             
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % SAR SIGNAL PROCESSING
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Range Reference Signal
            
            td_ref = t_fast-2*((Rc)/c);
            s_ref=exp(1j*pi*Kr*((td_ref.^2)));
            
            % Reference Signal in frequency domain
            s_ref_freq = fft(s_ref); 
            s_rx_freq = zeros(PRF * CPI, rbins); 
            
            % Range Compression
                                       
            for k=1:(length(t_slow))
              
                s_rx_freq(k,:)=fft(s_rx_noise(k,:)); % Range FFT
                s_mf_freq(k,:)=s_rx_freq(k,:).*conj(s_ref_freq);% Range Matched Filtering
                s_mf_time(k,:)=ifftshift(ifft(s_mf_freq(k,:))); % Range IFFT
            
            
            if strcmp(app.CoarseMotionCompensationDropDown.Value, 'Yes')
                s_mf_time(k,:) = s_mf_time(k,:).*exp(1j*4*pi*fc*ic*(del_R_all(k)));
            end
            end
            
            if strcmp(app.FineMotionCompensationDropDown.Value, 'Yes')
                % Cross correlation
                dr = c/(2*Bw);
                for q= 1:(length(t_slow))
                s_mf_time_rp(q,:) = abs(ifft(fft(s_mf_time(1,:)).*conj(fft(s_mf_time(q,:)))));
                peak_ccr(q) = find((max(s_mf_time_rp(q,:))== s_mf_time_rp(q,:)));%Find max. ind. (range shift) range)
                end
                S_peak_ccr = smooth((0:(length(t_slow))-1),peak_ccr,'loess');%smoothing the delays 
                RangeShifts = peak_ccr*dr;% range shifts
                SmRangeShifts = dr*S_peak_ccr;% smoothed range shifts
                
                fine_comp = exp(1j*4*pi*fc*ic*(SmRangeShifts));
                
                for b= 1:(length(t_slow))
                s_mf_time_ccr(b,:) = s_mf_time(b,:).*fine_comp(b);
                end 
            
            end
            
            % Azimuth FFT
            
            if strcmp(app.FineMotionCompensationDropDown.Value, 'Yes')
                for l=1:rbins
                    s_azi_f(:,l)=fftshift(fft(s_mf_time_ccr(:,l))); 
                end
            else
                for l=1:rbins
                    s_azi_f(:,l)=fftshift(fft(s_mf_time(:,l))); 
                end    
            end
            
            % Range Cell Migration Correction (RCMC)
            deltaR = (lambda^2 * Rc .* (Ka * (CPI / 2 - t_slow)).^2) / (8 * vp^2);  
            range_res = (c * ts) / 2;
            cells = round(deltaR / range_res); 
            rcm_max = max(cells); 
            rcmc=s_azi_f;
            for k=1:length(t_slow)
                for m=1:rbins-rcm_max
                    rcmc(k,m)=s_azi_f(k,m+cells(k));
                end
            end
            
            % Azimuth Reference Signal
            s_ref_azi_time=exp(-1j*pi*(Ka).*((0.5*CPI-t_slow)).^2);
            
            % Azimuth Matched Filter Spectrum
            s_ref_azi_freq=fftshift(fft(s_ref_azi_time)); 
            
            % Azimuth Compression
            for l=1:rbins
                sar_freq(:,l)=conj(rcmc(:,l)).*(s_ref_azi_freq); 
                % Azimuth IFFT 
                sar_time(:,l)=ifftshift(ifft(sar_freq(:,l))); 
            end
            
            % Normalization of received signal
            sar_time = (1/(length(t_slow)))*(1/rbins)*sar_time;
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % PLOTS
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            % Cross range axis (m)
            CR_max = CPI*vp;
            del_azi = La/2;
            azi_axis = (-CR_max/2:del_azi:CR_max/2-del_azi);
            % Range axis (m)
            range_axis = t_fast.*c.*0.5;


            % Two-dimensional SAR image in logscale
            sar_2d =20*log10(abs(sar_time));
            cmax = max(max(sar_2d));
            cmin = cmax-30;


            imagesc(app.UIAxes, range_axis * 1e-3, azi_axis, sar_2d,[cmin cmax]);
            xlabel(app.UIAxes, 'Range (Km)');
            ylabel(app.UIAxes, 'Azimuth (m)');
            title(app.UIAxes, sprintf('Slant Range of target=%.1f, Cross-range of target=%.1f \n', R_t, yt));
            app.UIAxes.YDir = 'normal';
            colorbar(app.UIAxes);
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
            %RESULTS
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            %1. RMSE
            [cross_range_index, range_index] = find(sar_2d==cmax);

            estimated_range = range_axis(range_index);
            estimated_cross_range = azi_axis(cross_range_index);
            ground_truth_range = R_t;
            ground_truth_cross_range = yt;

            rmse_range = rmse(estimated_range, ground_truth_range);
            app.RootMeanSquareErrorRangemEditField.Value = rmse_range;

            rmse_cross_range = rmse(estimated_cross_range, ground_truth_cross_range);
            app.RootMeanSquareErrorCrossrangemEditField.Value = rmse_cross_range;
             
            %2. PSLR
            main_lobe = sar_2d(cross_range_index, range_index);
            side_lobe = sar_2d(cross_range_index+1,range_index);          
             PSLR_dB = main_lobe-side_lobe;
            app.PeakSideLobeRatiodBEditField.Value = PSLR_dB;


            %3. ISLR
            sar_linear = abs(sar_time).^2;
            
            % Center pixel
            center = sar_linear(cross_range_index, range_index);
            
            % center + 4 neighbors
            up    = sar_linear(cross_range_index-1, range_index);
            down  = sar_linear(cross_range_index+1, range_index);
            left  = sar_linear(cross_range_index, range_index-1);
            right = sar_linear(cross_range_index, range_index+1);
            
            mainlobe_energy = mean(center + up + down + left + right);
            
            % Total energy in cross (row + column), excluding center
            row = sar_linear(cross_range_index, :);
            col = sar_linear(:, range_index);
            
            row_energy = sum(row) - (center);
            col_energy = sum(col) - (center);
            
            cross_energy = row_energy + col_energy;
            
            % Remove the 4 neighbors already counted in mainlobe
            sidelobe_energy = mean(cross_energy - (up + down + left + right));
            
            ISLR =  mainlobe_energy/sidelobe_energy;
            ISLR_dB = 10 * log10(ISLR);

            app.IntegratedSideLobeRatioEditField.Value = ISLR_dB;

            %4. Entropy
            E = entropy(abs(sar_time));
            app.EntropyEditField.Value = E;

            %5. Contrast
            T_squared = abs(sar_2d).^2;
            mean_T_squared = mean(T_squared(:));            
            contrast_numerator = sqrt(mean((T_squared(:) - mean_T_squared).^2));            
            Contrast = contrast_numerator / mean_T_squared;
            app.ContrastEditField.Value = Contrast;

            

        end
      end




    % Callbacks that handle component events
    methods (Access = private)

        % Button pushed function: RunButton
        function RunButtonPushed(app, event)
            % INPUT PARAMETERS FOR SAR
            % Indraprastha Institute of Information Technology

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % PLATFORM PARAMETERS
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Low squint angle (degrees)            
            theta_sq = app.SquintAngledegreesEditField.Value;
            % Altitude (m)
            z_0 = app.AltitudemEditField.Value;
            % Range distance to center of swath area (m)
            xc = app.DistancetocenterofswarthmEditField.Value; 
            %Mean Velocity of platform (along track - component)(m/s)
            vp = app.PlatformvelocitymsEditField.Value; 
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Errors
            
            % Translational motion errors
            if app.TranslationalErrorCheckBox.Value
                trans_err_flag = 1;
            else
                trans_err_flag = 0;
            end
            %Standard Deviation of Velocity of platform (along track - component) with error(m/s)
            std_vp = app.AzimuthalongtrackerrormEditField.Value;
            %Mean Velocity of platform (cross track azimuth - component)(m/s)
            vx = 0;
            %Standard Deviation of Velocity of platform (cross track azimuth - component) with error(m/s)
            std_vx = app.CrosstrackerrormEditField.Value;
            %Mean Velocity of platform (cross track elevation - component)(m/s)
            vz = 0;
            %Standard Velocity of platform (cross track elevation - component) with error(m/s)
            std_vz = app.ElevationerrormEditField.Value;
            
            % Rotational Motion Errors
            if app.RotationalErrorCheckBox.Value
                rot_err_flag = 1;
            else
                rot_err_flag = 0;
            end
            %Mean of Yaw angle (degrees)
            yaw = app.YawdegreesEditField.Value;
            % Standard Deviation of Yaw Angle
            std_yaw = 0;
            %Pitch angle (degrees)
            pitch = app.PitchdegreesEditField.Value;
            % Standard Deviation of Pitch Angle
            std_pitch = 0;
            %RCS (dbsm)
            RCS = app.RadarCrossSectiondBsmEditField.Value;
            %Noise Type
            selectedNoise = app.NoiseTypeDropDown.Value;

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % RADAR PARAMETERS
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %Carrier frequency (Hz)
            fc = 1e9*app.CarrierFrequencyGHzEditField.Value;
            %Baseband bandwidth (Hz)
            Bw = 1e6*app.BandwidthMHzEditField.Value;
            %Pulse Repetition Frequency (Hz)
            PRF = app.PulseRepitionFrequencyHzEditField.Value;
            %Total time (CPI) (sec)
            CPI = app.CoherentProcessingIntervalsEditField.Value;
            %Antenna length actual (m)
            La = 2*vp*(1/PRF);
            %Duty Cycle
            DC = app.DutyCycleEditField.Value;
            %Chirp Pulse Duration
            Tp = DC/PRF;
            % Signal to noise ratio (dB)
            SNR = app.SignaltoNoiseRatiodBEditField.Value;
            % Range envelope
            w_r = 1;
            % Azimuth envelope
            w_a = 1;
            % Half sawth width (Target is located within [Xc-X0,Xc+X0])
            R0 = app.HalfSwathWidthalongtrackmEditField.Value; 
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %TARGET PARAMETERS
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Number of targets
            ntarget= 1;
            %Range of target (m)
            xt = app.CrosstrackmEditField.Value;
            %Cross range (m)
            yt = app.AzimuthalongtrackmEditField.Value;
            % Height of target
            zt = app.ElevationmEditField.Value;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % MOTION COMPENSATION
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Coarse motion compensation
            
            % image type
            
            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % GLOBAL CONSTANTS
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Propagation speed
            c = 3e8;
            % Propagation frequency
            ic = 1 / c;
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % DERIVED VARIABLES
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Wavelength (m)
            lambda = c / fc; 
            % Range Chirp Rate (Hz/s)
            Kr = Bw / Tp; 
            % Slant range to center of swath area (m)
            Rc = sqrt(z_0^2 + xc^2);
            % Slant range (m)
            R_t = sqrt((z_0^2) + (xt^2));
            % Azimuth parameter Linear Azimuth FM rate
            Ka = (2 * vp^2) / (lambda * Rc);
            % Time Domain Sampling Interval (sec)
            ts = 1 / (2 * Bw);
            % Start time of sampling (sec)
            Ts = (2 * (Rc - R0)) / c; 
            % End time of sampling (sec)
            Tf = (2 * (Rc + R0)) / c;
            % Number of time (Range) samples)
            rbins = ceil(((Tf - Ts)) / ts); 
            t_fast = Ts + (0:rbins -1) * ts; 
            %Slow Time Array
            t_slow = linspace(0, CPI, PRF * CPI)'; 
            % Actual velocity of platform (while incorporating errors)
            vp_e = std_vp .* randn(length(t_slow), 1) + vp; 
            vx_e = std_vx .* randn(length(t_slow), 1) + vx; 
            vz_e = std_vz .* randn(length(t_slow), 1) + vz;
            % Rotational motion errors
            yaw_e = std_yaw .* ones(length(t_slow), 1) + yaw;
            pitch_e = std_pitch .* ones(2000, 1) + pitch;

            % Global constants
            y_0 = 0;
            x_0 = 0;
 
            %Dynamic Error Flags
            
            if (CPI < (1/PRF))
                msg = "error: Coherent Processing Interval < Pulse Repetition Interval";
            elseif (R0 > xc)
                msg = "error: Half swath width alongtrack > Distance to the centre of swath";
            elseif (std_vp > vp)
                msg = "error: Azimuth alongtrack error > Platform velocity";
            elseif (std_vx > vp)
                msg = "error: Crosstrack error > Platform velocity";
            elseif (std_vz > vp)
                msg = "error: Elevation error > Platform velocity";
            elseif (xt < xc - R0)
                msg = "error: Crosstrack < Distance to the centre of swath - Half swath width alongtrack";
            elseif (xt > xc + R0)
                msg = "error: Crosstrack > Distance to the centre of swath + Half swath width alongtrack";
            elseif (yt < -(0.5 * CPI * vp))
                msg = "error: Azimuth alongtrack < -(0.5*Coherent Processing Interval*Platform velocity)";
            elseif (yt > (0.5 * CPI * vp))
                msg = "error: Azimuth alongtrack > (0.5*Coherent Processing Interval*Platform velocity)";
            elseif (zt > z_0)
                msg = "error: Elevation crosstrack > Altitude";
            else
                msg ="";
                app.runSAR();

            end
            
            % Set the message
            app.ConsoleTextArea.Value = msg;
            
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Get the file path for locating images
            pathToMLAPP = fileparts(mfilename('fullpath'));

            % Create UIFigure and hide until all components are created
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Position = [100 50 1279 810];
            app.UIFigure.Name = 'MATLAB App';

            % Create TabGroup
            app.TabGroup = uitabgroup(app.UIFigure);
            app.TabGroup.Position = [1 0 1279 812];

            % Create ParametersTab
            app.ParametersTab = uitab(app.TabGroup);
            app.ParametersTab.Title = 'Parameters';

            % Create UIAxes
            app.UIAxes = uiaxes(app.ParametersTab);
            title(app.UIAxes, {'SAR'; ''})
            xlabel(app.UIAxes, 'X')
            ylabel(app.UIAxes, 'Y')
            zlabel(app.UIAxes, 'Z')
            app.UIAxes.Position = [372 271 534 379];

            % Create SARMotionCompensationAnalysisforPointTargetsLabel
            app.SARMotionCompensationAnalysisforPointTargetsLabel = uilabel(app.ParametersTab);
            app.SARMotionCompensationAnalysisforPointTargetsLabel.BackgroundColor = [0.2471 0.6784 0.6588];
            app.SARMotionCompensationAnalysisforPointTargetsLabel.HorizontalAlignment = 'center';
            app.SARMotionCompensationAnalysisforPointTargetsLabel.FontName = 'Arial';
            app.SARMotionCompensationAnalysisforPointTargetsLabel.FontSize = 35;
            app.SARMotionCompensationAnalysisforPointTargetsLabel.FontWeight = 'bold';
            app.SARMotionCompensationAnalysisforPointTargetsLabel.FontColor = [1 1 1];
            app.SARMotionCompensationAnalysisforPointTargetsLabel.Position = [255 731 1003 48];
            app.SARMotionCompensationAnalysisforPointTargetsLabel.Text = '  SAR: Motion Compensation Analysis for Point Targets ';

            % Create RadarCrossSectiondBsmEditFieldLabel
            app.RadarCrossSectiondBsmEditFieldLabel = uilabel(app.ParametersTab);
            app.RadarCrossSectiondBsmEditFieldLabel.HorizontalAlignment = 'right';
            app.RadarCrossSectiondBsmEditFieldLabel.Position = [46 29 158 22];
            app.RadarCrossSectiondBsmEditFieldLabel.Text = 'Radar Cross Section (dBsm)';

            % Create RadarCrossSectiondBsmEditField
            app.RadarCrossSectiondBsmEditField = uieditfield(app.ParametersTab, 'numeric');
            app.RadarCrossSectiondBsmEditField.Limits = [0 20];
            app.RadarCrossSectiondBsmEditField.Position = [219 29 100 22];

            % Create ElevationmEditFieldLabel
            app.ElevationmEditFieldLabel = uilabel(app.ParametersTab);
            app.ElevationmEditFieldLabel.HorizontalAlignment = 'right';
            app.ElevationmEditFieldLabel.Position = [128 65 76 22];
            app.ElevationmEditFieldLabel.Text = 'Elevation (m)';

            % Create ElevationmEditField
            app.ElevationmEditField = uieditfield(app.ParametersTab, 'numeric');
            app.ElevationmEditField.Position = [219 65 100 22];

            % Create AzimuthalongtrackmEditFieldLabel
            app.AzimuthalongtrackmEditFieldLabel = uilabel(app.ParametersTab);
            app.AzimuthalongtrackmEditFieldLabel.HorizontalAlignment = 'right';
            app.AzimuthalongtrackmEditFieldLabel.Position = [75 101 129 22];
            app.AzimuthalongtrackmEditFieldLabel.Text = 'Azimuth alongtrack (m)';

            % Create AzimuthalongtrackmEditField
            app.AzimuthalongtrackmEditField = uieditfield(app.ParametersTab, 'numeric');
            app.AzimuthalongtrackmEditField.Position = [219 101 100 22];
            app.AzimuthalongtrackmEditField.Value = 10;

            % Create CrosstrackmEditFieldLabel
            app.CrosstrackmEditFieldLabel = uilabel(app.ParametersTab);
            app.CrosstrackmEditFieldLabel.HorizontalAlignment = 'right';
            app.CrosstrackmEditFieldLabel.Position = [120 137 84 22];
            app.CrosstrackmEditFieldLabel.Text = 'Crosstrack (m)';

            % Create CrosstrackmEditField
            app.CrosstrackmEditField = uieditfield(app.ParametersTab, 'numeric');
            app.CrosstrackmEditField.Position = [219 137 100 22];
            app.CrosstrackmEditField.Value = 6116;

            % Create TargetParametersLabel
            app.TargetParametersLabel = uilabel(app.ParametersTab);
            app.TargetParametersLabel.BackgroundColor = [0.4 0.4 0.4];
            app.TargetParametersLabel.HorizontalAlignment = 'center';
            app.TargetParametersLabel.FontSize = 15;
            app.TargetParametersLabel.FontWeight = 'bold';
            app.TargetParametersLabel.FontColor = [1 1 1];
            app.TargetParametersLabel.Position = [21 173 298 22];
            app.TargetParametersLabel.Text = 'Target Parameters';

            % Create YawdegreesEditFieldLabel
            app.YawdegreesEditFieldLabel = uilabel(app.ParametersTab);
            app.YawdegreesEditFieldLabel.HorizontalAlignment = 'right';
            app.YawdegreesEditFieldLabel.Position = [1058 470 83 22];
            app.YawdegreesEditFieldLabel.Text = 'Yaw (degrees)';

            % Create YawdegreesEditField
            app.YawdegreesEditField = uieditfield(app.ParametersTab, 'numeric');
            app.YawdegreesEditField.Limits = [0 360];
            app.YawdegreesEditField.Position = [1156 470 100 22];
            app.YawdegreesEditField.Value = 10;

            % Create PitchdegreesEditFieldLabel
            app.PitchdegreesEditFieldLabel = uilabel(app.ParametersTab);
            app.PitchdegreesEditFieldLabel.HorizontalAlignment = 'right';
            app.PitchdegreesEditFieldLabel.Position = [1055 506 87 22];
            app.PitchdegreesEditFieldLabel.Text = 'Pitch (degrees)';

            % Create PitchdegreesEditField
            app.PitchdegreesEditField = uieditfield(app.ParametersTab, 'numeric');
            app.PitchdegreesEditField.Limits = [0 180];
            app.PitchdegreesEditField.Position = [1157 506 100 22];
            app.PitchdegreesEditField.Value = 50;

            % Create RolldegreesEditFieldLabel
            app.RolldegreesEditFieldLabel = uilabel(app.ParametersTab);
            app.RolldegreesEditFieldLabel.HorizontalAlignment = 'right';
            app.RolldegreesEditFieldLabel.Position = [1061 542 81 22];
            app.RolldegreesEditFieldLabel.Text = 'Roll (degrees)';

            % Create RolldegreesEditField
            app.RolldegreesEditField = uieditfield(app.ParametersTab, 'numeric');
            app.RolldegreesEditField.Position = [1157 542 100 22];

            % Create ElevationerrormEditFieldLabel
            app.ElevationerrormEditFieldLabel = uilabel(app.ParametersTab);
            app.ElevationerrormEditFieldLabel.HorizontalAlignment = 'right';
            app.ElevationerrormEditFieldLabel.Position = [1037 579 105 22];
            app.ElevationerrormEditFieldLabel.Text = 'Elevation error (m)';

            % Create ElevationerrormEditField
            app.ElevationerrormEditField = uieditfield(app.ParametersTab, 'numeric');
            app.ElevationerrormEditField.Position = [1157 579 100 22];
            app.ElevationerrormEditField.Value = 1;

            % Create AzimuthalongtrackerrormEditFieldLabel
            app.AzimuthalongtrackerrormEditFieldLabel = uilabel(app.ParametersTab);
            app.AzimuthalongtrackerrormEditFieldLabel.HorizontalAlignment = 'right';
            app.AzimuthalongtrackerrormEditFieldLabel.Position = [984 616 158 22];
            app.AzimuthalongtrackerrormEditFieldLabel.Text = 'Azimuth alongtrack error (m)';

            % Create AzimuthalongtrackerrormEditField
            app.AzimuthalongtrackerrormEditField = uieditfield(app.ParametersTab, 'numeric');
            app.AzimuthalongtrackerrormEditField.Position = [1157 616 100 22];
            app.AzimuthalongtrackerrormEditField.Value = 0.5;

            % Create CrosstrackerrormEditFieldLabel
            app.CrosstrackerrormEditFieldLabel = uilabel(app.ParametersTab);
            app.CrosstrackerrormEditFieldLabel.HorizontalAlignment = 'right';
            app.CrosstrackerrormEditFieldLabel.Position = [1029 653 113 22];
            app.CrosstrackerrormEditFieldLabel.Text = 'Crosstrack error (m)';

            % Create CrosstrackerrormEditField
            app.CrosstrackerrormEditField = uieditfield(app.ParametersTab, 'numeric');
            app.CrosstrackerrormEditField.Position = [1157 653 100 22];
            app.CrosstrackerrormEditField.Value = 1;

            % Create TranslationalandRotationalMotionErrorsLabel
            app.TranslationalandRotationalMotionErrorsLabel = uilabel(app.ParametersTab);
            app.TranslationalandRotationalMotionErrorsLabel.BackgroundColor = [0.4 0.4 0.4];
            app.TranslationalandRotationalMotionErrorsLabel.HorizontalAlignment = 'center';
            app.TranslationalandRotationalMotionErrorsLabel.FontSize = 15;
            app.TranslationalandRotationalMotionErrorsLabel.FontWeight = 'bold';
            app.TranslationalandRotationalMotionErrorsLabel.FontColor = [1 1 1];
            app.TranslationalandRotationalMotionErrorsLabel.Position = [950 688 309 22];
            app.TranslationalandRotationalMotionErrorsLabel.Text = 'Translational and Rotational Motion Errors';

            % Create DutyCycleEditFieldLabel
            app.DutyCycleEditFieldLabel = uilabel(app.ParametersTab);
            app.DutyCycleEditFieldLabel.HorizontalAlignment = 'right';
            app.DutyCycleEditFieldLabel.Position = [117 247 86 22];
            app.DutyCycleEditFieldLabel.Text = 'Duty Cycle (%)';

            % Create DutyCycleEditField
            app.DutyCycleEditField = uieditfield(app.ParametersTab, 'numeric');
            app.DutyCycleEditField.Limits = [-Inf 51];
            app.DutyCycleEditField.Position = [218 247 100 22];
            app.DutyCycleEditField.Value = 0.005;

            % Create SignaltoNoiseRatiodBEditFieldLabel
            app.SignaltoNoiseRatiodBEditFieldLabel = uilabel(app.ParametersTab);
            app.SignaltoNoiseRatiodBEditFieldLabel.HorizontalAlignment = 'right';
            app.SignaltoNoiseRatiodBEditFieldLabel.Position = [59 283 144 22];
            app.SignaltoNoiseRatiodBEditFieldLabel.Text = 'Signal to Noise Ratio (dB)';

            % Create SignaltoNoiseRatiodBEditField
            app.SignaltoNoiseRatiodBEditField = uieditfield(app.ParametersTab, 'numeric');
            app.SignaltoNoiseRatiodBEditField.Limits = [-20 100];
            app.SignaltoNoiseRatiodBEditField.Position = [218 283 100 22];

            % Create SquintAngledegreesEditFieldLabel
            app.SquintAngledegreesEditFieldLabel = uilabel(app.ParametersTab);
            app.SquintAngledegreesEditFieldLabel.HorizontalAlignment = 'right';
            app.SquintAngledegreesEditFieldLabel.Position = [75 398 128 22];
            app.SquintAngledegreesEditFieldLabel.Text = 'Squint Angle (degrees)';

            % Create SquintAngledegreesEditField
            app.SquintAngledegreesEditField = uieditfield(app.ParametersTab, 'numeric');
            app.SquintAngledegreesEditField.Limits = [-Inf 10];
            app.SquintAngledegreesEditField.Position = [218 398 100 22];
            app.SquintAngledegreesEditField.Value =0;

            % Create BandwidthMHzEditFieldLabel
            app.BandwidthMHzEditFieldLabel = uilabel(app.ParametersTab);
            app.BandwidthMHzEditFieldLabel.HorizontalAlignment = 'right';
            app.BandwidthMHzEditFieldLabel.Position = [105 506 98 22];
            app.BandwidthMHzEditFieldLabel.Text = 'Bandwidth (MHz)';

            % Create BandwidthMHzEditField
            app.BandwidthMHzEditField = uieditfield(app.ParametersTab, 'numeric');
            app.BandwidthMHzEditField.Position = [218 506 100 22];
            app.BandwidthMHzEditField.Value = 100;

            % Create CarrierFrequencyGHzEditFieldLabel
            app.CarrierFrequencyGHzEditFieldLabel = uilabel(app.ParametersTab);
            app.CarrierFrequencyGHzEditFieldLabel.HorizontalAlignment = 'right';
            app.CarrierFrequencyGHzEditFieldLabel.Position = [65 542 138 22];
            app.CarrierFrequencyGHzEditFieldLabel.Text = 'Carrier Frequency (GHz)';

            % Create CarrierFrequencyGHzEditField
            app.CarrierFrequencyGHzEditField = uieditfield(app.ParametersTab, 'numeric');
            app.CarrierFrequencyGHzEditField.Position = [218 542 100 22];
            app.CarrierFrequencyGHzEditField.Value = 5;

            % Create PlatformvelocitymsEditFieldLabel
            app.PlatformvelocitymsEditFieldLabel = uilabel(app.ParametersTab);
            app.PlatformvelocitymsEditFieldLabel.HorizontalAlignment = 'right';
            app.PlatformvelocitymsEditFieldLabel.Position = [79 616 124 22];
            app.PlatformvelocitymsEditFieldLabel.Text = 'Platform velocity (m/s)';

            % Create PlatformvelocitymsEditField
            app.PlatformvelocitymsEditField = uieditfield(app.ParametersTab, 'numeric');
            app.PlatformvelocitymsEditField.Limits = [100 Inf];
            app.PlatformvelocitymsEditField.Position = [218 616 100 22];
            app.PlatformvelocitymsEditField.Value = 400;

            % Create AltitudemEditFieldLabel
            app.AltitudemEditFieldLabel = uilabel(app.ParametersTab);
            app.AltitudemEditFieldLabel.HorizontalAlignment = 'right';
            app.AltitudemEditFieldLabel.Position = [136 653 67 22];
            app.AltitudemEditFieldLabel.Text = 'Altitude (m)';

            % Create AltitudemEditField
            app.AltitudemEditField = uieditfield(app.ParametersTab, 'numeric');
            app.AltitudemEditField.Limits = [1000 Inf];
            app.AltitudemEditField.Position = [218 653 100 22];
            app.AltitudemEditField.Value = 8000;

            % Create PulseRepitionFrequencyHzEditFieldLabel
            app.PulseRepitionFrequencyHzEditFieldLabel = uilabel(app.ParametersTab);
            app.PulseRepitionFrequencyHzEditFieldLabel.HorizontalAlignment = 'right';
            app.PulseRepitionFrequencyHzEditFieldLabel.Position = [34 470 169 22];
            app.PulseRepitionFrequencyHzEditFieldLabel.Text = 'Pulse Repition Frequency (Hz)';

            % Create PulseRepitionFrequencyHzEditField
            app.PulseRepitionFrequencyHzEditField = uieditfield(app.ParametersTab, 'numeric');
            app.PulseRepitionFrequencyHzEditField.Position = [218 470 100 22];
            app.PulseRepitionFrequencyHzEditField.Value = 1000;

            % Create CoherentProcessingIntervalsEditFieldLabel
            app.CoherentProcessingIntervalsEditFieldLabel = uilabel(app.ParametersTab);
            app.CoherentProcessingIntervalsEditFieldLabel.HorizontalAlignment = 'right';
            app.CoherentProcessingIntervalsEditFieldLabel.Position = [25 434 178 22];
            app.CoherentProcessingIntervalsEditFieldLabel.Text = 'Coherent Processing Interval (s)';

            % Create CoherentProcessingIntervalsEditField
            app.CoherentProcessingIntervalsEditField = uieditfield(app.ParametersTab, 'numeric');
            app.CoherentProcessingIntervalsEditField.Position = [218 434 100 22];
            app.CoherentProcessingIntervalsEditField.Value = 2;

            % Create DistancetocenterofswarthmEditFieldLabel
            app.DistancetocenterofswarthmEditFieldLabel = uilabel(app.ParametersTab);
            app.DistancetocenterofswarthmEditFieldLabel.HorizontalAlignment = 'right';
            app.DistancetocenterofswarthmEditFieldLabel.Position = [27 579 176 22];
            app.DistancetocenterofswarthmEditFieldLabel.Text = 'Distance to center of swarth (m)';

            % Create DistancetocenterofswarthmEditField
            app.DistancetocenterofswarthmEditField = uieditfield(app.ParametersTab, 'numeric');
            app.DistancetocenterofswarthmEditField.Limits = [0 Inf];
            app.DistancetocenterofswarthmEditField.Position = [218 579 100 22];
            app.DistancetocenterofswarthmEditField.Value = 6000;

            % Create HalfSwathWidthalongtrackmEditFieldLabel
            app.HalfSwathWidthalongtrackmEditFieldLabel = uilabel(app.ParametersTab);
            app.HalfSwathWidthalongtrackmEditFieldLabel.HorizontalAlignment = 'right';
            app.HalfSwathWidthalongtrackmEditFieldLabel.Position = [23 209 181 22];
            app.HalfSwathWidthalongtrackmEditFieldLabel.Text = 'Half Swath Width along track (m)';

            % Create HalfSwathWidthalongtrackmEditField
            app.HalfSwathWidthalongtrackmEditField = uieditfield(app.ParametersTab, 'numeric');
            app.HalfSwathWidthalongtrackmEditField.Position = [219 209 100 22];
            app.HalfSwathWidthalongtrackmEditField.Value = 500;

            % Create RadarandRadarPlatformParametersLabel
            app.RadarandRadarPlatformParametersLabel = uilabel(app.ParametersTab);
            app.RadarandRadarPlatformParametersLabel.BackgroundColor = [0.4 0.4 0.4];
            app.RadarandRadarPlatformParametersLabel.HorizontalAlignment = 'center';
            app.RadarandRadarPlatformParametersLabel.FontSize = 15;
            app.RadarandRadarPlatformParametersLabel.FontWeight = 'bold';
            app.RadarandRadarPlatformParametersLabel.FontColor = [1 1 1];
            app.RadarandRadarPlatformParametersLabel.Position = [21 688 298 22];
            app.RadarandRadarPlatformParametersLabel.Text = 'Radar and Radar Platform Parameters';

            % Create RunButton
            app.RunButton = uibutton(app.ParametersTab, 'push');
            app.RunButton.ButtonPushedFcn = createCallbackFcn(app, @RunButtonPushed, true);
            app.RunButton.Icon = fullfile(pathToMLAPP, 'run.png');
            app.RunButton.BackgroundColor = [0.9412 0.9412 0.9412];
            app.RunButton.FontSize = 16;
            app.RunButton.Position = [1040 65 119 32];
            app.RunButton.Text = 'Run';

            % Create Image
            app.Image = uiimage(app.ParametersTab);
            app.Image.Position = [21 731 235 48];
            app.Image.ImageSource = fullfile(pathToMLAPP, 'iiitd.jpg');

            % Create ContrastEditFieldLabel
            app.ContrastEditFieldLabel = uilabel(app.ParametersTab);
            app.ContrastEditFieldLabel.HorizontalAlignment = 'right';
            app.ContrastEditFieldLabel.Position = [1091 210 51 22];
            app.ContrastEditFieldLabel.Text = 'Contrast';

            % Create ContrastEditField
            app.ContrastEditField = uieditfield(app.ParametersTab, 'numeric');
            app.ContrastEditField.Position = [1157 210 100 22];
            app.ContrastEditField.Value = -1;

            % Create EntropyEditFieldLabel
            app.EntropyEditFieldLabel = uilabel(app.ParametersTab);
            app.EntropyEditFieldLabel.HorizontalAlignment = 'right';
            app.EntropyEditFieldLabel.Position = [1095 246 47 22];
            app.EntropyEditFieldLabel.Text = 'Entropy';

            % Create EntropyEditField
            app.EntropyEditField = uieditfield(app.ParametersTab, 'numeric');
            app.EntropyEditField.Position = [1157 246 100 22];
            app.EntropyEditField.Value = -1;

            % Create IntegratedSideLobeRatioEditFieldLabel
            app.IntegratedSideLobeRatioEditFieldLabel = uilabel(app.ParametersTab);
            app.IntegratedSideLobeRatioEditFieldLabel.HorizontalAlignment = 'right';
            app.IntegratedSideLobeRatioEditFieldLabel.Position = [994 283 148 22];
            app.IntegratedSideLobeRatioEditFieldLabel.Text = 'Integrated Side Lobe Ratio';

            % Create IntegratedSideLobeRatioEditField
            app.IntegratedSideLobeRatioEditField = uieditfield(app.ParametersTab, 'numeric');
            app.IntegratedSideLobeRatioEditField.Position = [1157 283 100 22];
            app.IntegratedSideLobeRatioEditField.Value = -1;

            % Create PeakSideLobeRatiodBEditFieldLabel
            app.PeakSideLobeRatiodBEditFieldLabel = uilabel(app.ParametersTab);
            app.PeakSideLobeRatiodBEditFieldLabel.HorizontalAlignment = 'right';
            app.PeakSideLobeRatiodBEditFieldLabel.Position = [995 320 147 22];
            app.PeakSideLobeRatiodBEditFieldLabel.Text = 'Peak Side Lobe Ratio (dB)';

            % Create PeakSideLobeRatiodBEditField
            app.PeakSideLobeRatiodBEditField = uieditfield(app.ParametersTab, 'numeric');
            app.PeakSideLobeRatiodBEditField.Position = [1157 320 100 22];
            app.PeakSideLobeRatiodBEditField.Value = -1;

            % Create RootMeanSquareErrorRangemEditFieldLabel
            app.RootMeanSquareErrorRangemEditFieldLabel = uilabel(app.ParametersTab);
            app.RootMeanSquareErrorRangemEditFieldLabel.HorizontalAlignment = 'right';
            app.RootMeanSquareErrorRangemEditFieldLabel.Position = [945 399 197 22];
            app.RootMeanSquareErrorRangemEditFieldLabel.Text = 'Root Mean Square Error Range (m)';

            % Create RootMeanSquareErrorRangemEditField
            app.RootMeanSquareErrorRangemEditField = uieditfield(app.ParametersTab, 'numeric');
            app.RootMeanSquareErrorRangemEditField.Position = [1157 399 100 22];
            app.RootMeanSquareErrorRangemEditField.Value = -1;

            % Create ResultsandMetricLabel
            app.ResultsandMetricLabel = uilabel(app.ParametersTab);
            app.ResultsandMetricLabel.BackgroundColor = [0.4 0.4 0.4];
            app.ResultsandMetricLabel.HorizontalAlignment = 'center';
            app.ResultsandMetricLabel.FontSize = 15;
            app.ResultsandMetricLabel.FontWeight = 'bold';
            app.ResultsandMetricLabel.FontColor = [1 1 1];
            app.ResultsandMetricLabel.Position = [951 434 308 22];
            app.ResultsandMetricLabel.Text = 'Results and Metric';

            % Create NoiseTypeDropDownLabel
            app.NoiseTypeDropDownLabel = uilabel(app.ParametersTab);
            app.NoiseTypeDropDownLabel.HorizontalAlignment = 'right';
            app.NoiseTypeDropDownLabel.Position = [79 362 65 22];
            app.NoiseTypeDropDownLabel.Text = 'Noise Type';

            % Create NoiseTypeDropDown
            app.NoiseTypeDropDown = uidropdown(app.ParametersTab);
            app.NoiseTypeDropDown.Items = {'White Gaussian Noise', 'Speckle Noise'};
            app.NoiseTypeDropDown.Position = [159 362 160 22];
            app.NoiseTypeDropDown.Value = 'White Gaussian Noise';

            % Create ConsoleTextAreaLabel
            app.ConsoleTextAreaLabel = uilabel(app.ParametersTab);
            app.ConsoleTextAreaLabel.FontSize = 16;
            app.ConsoleTextAreaLabel.Position = [372 70 64 22];
            app.ConsoleTextAreaLabel.Text = 'Console';

            % Create ConsoleTextArea
            app.ConsoleTextArea = uitextarea(app.ParametersTab);
            app.ConsoleTextArea.FontSize = 16;
            app.ConsoleTextArea.FontColor = [1 0 0];
            app.ConsoleTextArea.Position = [449 65 582 32];

            % Create RootMeanSquareErrorCrossrangemEditFieldLabel
            app.RootMeanSquareErrorCrossrangemEditFieldLabel = uilabel(app.ParametersTab);
            app.RootMeanSquareErrorCrossrangemEditFieldLabel.HorizontalAlignment = 'right';
            app.RootMeanSquareErrorCrossrangemEditFieldLabel.Position = [917 362 225 22];
            app.RootMeanSquareErrorCrossrangemEditFieldLabel.Text = 'Root Mean Square Error Crossrange (m)';

            % Create RootMeanSquareErrorCrossrangemEditField
            app.RootMeanSquareErrorCrossrangemEditField = uieditfield(app.ParametersTab, 'numeric');
            app.RootMeanSquareErrorCrossrangemEditField.Position = [1157 362 100 22];
            app.RootMeanSquareErrorCrossrangemEditField.Value = -1;

            % Create CoarseMotionCompensationDropDownLabel
            app.CoarseMotionCompensationDropDownLabel = uilabel(app.ParametersTab);
            app.CoarseMotionCompensationDropDownLabel.HorizontalAlignment = 'right';
            app.CoarseMotionCompensationDropDownLabel.FontWeight = 'bold';
            app.CoarseMotionCompensationDropDownLabel.Position = [372 173 175 22];
            app.CoarseMotionCompensationDropDownLabel.Text = 'Coarse Motion Compensation';

            % Create CoarseMotionCompensationDropDown
            app.CoarseMotionCompensationDropDown = uidropdown(app.ParametersTab);
            app.CoarseMotionCompensationDropDown.Items = {'Yes', 'No'};
            app.CoarseMotionCompensationDropDown.Position = [562 173 100 22];
            app.CoarseMotionCompensationDropDown.Value = 'Yes';

            % Create FineMotionCompensationDropDownLabel
            app.FineMotionCompensationDropDownLabel = uilabel(app.ParametersTab);
            app.FineMotionCompensationDropDownLabel.HorizontalAlignment = 'right';
            app.FineMotionCompensationDropDownLabel.FontWeight = 'bold';
            app.FineMotionCompensationDropDownLabel.Position = [679 173 159 22];
            app.FineMotionCompensationDropDownLabel.Text = 'Fine Motion Compensation';

            % Create FineMotionCompensationDropDown
            app.FineMotionCompensationDropDown = uidropdown(app.ParametersTab);
            app.FineMotionCompensationDropDown.Items = {'Yes', 'No'};
            app.FineMotionCompensationDropDown.Position = [853 173 100 22];
            app.FineMotionCompensationDropDown.Value = 'Yes';

            % Create TranslationalErrorCheckBox
            app.TranslationalErrorCheckBox = uicheckbox(app.ParametersTab);
            app.TranslationalErrorCheckBox.Text = 'Translational Error';
            app.TranslationalErrorCheckBox.FontWeight = 'bold';
            app.TranslationalErrorCheckBox.Position = [419 210 129 22];
            app.TranslationalErrorCheckBox.Value = true;

            % Create RotationalErrorCheckBox
            app.RotationalErrorCheckBox = uicheckbox(app.ParametersTab);
            app.RotationalErrorCheckBox.Text = 'Rotational Error';
            app.RotationalErrorCheckBox.FontWeight = 'bold';
            app.RotationalErrorCheckBox.Position = [724 208 114 22];
            app.RotationalErrorCheckBox.Value = true;

            % Create ReferenceTab
            app.ReferenceTab = uitab(app.TabGroup);
            app.ReferenceTab.Title = 'Reference';

            % Create Image2
            app.Image2 = uiimage(app.ReferenceTab);
            app.Image2.Position = [21 95 631 615];
            app.Image2.ImageSource = fullfile(pathToMLAPP, 'System_model.png');

            % Create Image3
            app.Image3 = uiimage(app.ReferenceTab);
            app.Image3.Position = [665 65 591 613];
            app.Image3.ImageSource = fullfile(pathToMLAPP, 'Motion_errors.png');

            % Show the figure after all components are created
            app.UIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = SAR_new_exported

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.UIFigure)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.UIFigure)
        end
    end
end