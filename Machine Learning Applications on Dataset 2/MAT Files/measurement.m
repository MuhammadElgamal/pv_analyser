classdef measurement
    properties
        conditioned_current;
        measured_voltage;
        simulated_voltage;
        measuring_time;
        temperature;
        irradiance;
        conditioned_voltage;
        measured_current;
        simulated_current;
        panel_id;
        vmin;
        vmax;
        isc;
        voc;
        mp;
        imp;
        vmp;
        fillfac;
        efficiency;
        rs;
        rp;
        idark;
        iph;
        ideality;
        point_count;
        voltage_step;
        voltage_step_variation;
        error_extraction;   % error between the extracted parameters and actual panel

        model_found=true;
        valid_conditioning = false;
        %% Cell technological perfromance
        cell_voc ;
        cell_jsc;
        cell_jmp;
        cell_vmp;

        %% Checking validity of parameters
        mp_expected;
        isc_expected;
        voc_expected;
        fillfac_expected;
        effciency_expected;
        rs_expected;
        rp_expected;
        idark_expected;
        ideality_expected;
        iph_expected;
        voltage_expected;
        current_expected;
        error_expectation;
        imp_expected;
        vmp_expected;
        %% Validity variables
        valid_isc;
        valid_voc;
        valid_fillfac;
        valid_efficiency;
        valid_error;
        %% stornelli_parameters
        stornelli_rs;
        stornelli_rp;
        stornelli_idark;
        stornelli_iph;
        stornelli_ideality;
        stornelli_error;
        stornelli_voltage;
        stornelli_current;
    end
    methods
        function reading=read_file(reading, number, first_report)
            if not(first_report)
                name = "reading no" + number +".mat";
            else
                name = "report1 reading no" + number +".mat";
            end
            load(name);
            reading.conditioned_current = conditioned_current;
            reading.measured_voltage=measured_voltage;
            reading.simulated_voltage=simulated_voltage;
            t = measuring_time;
            reading.measuring_time=datetime(t(1), t(2), t(3), t(4), t(5), t(6));
            reading.temperature=temperature;
            reading.irradiance=irradiance;
            reading.conditioned_current=conditioned_current;
            reading.conditioned_voltage=conditioned_voltage;
            reading.measured_current=measured_current;
            reading.measured_voltage=measured_voltage;
            reading.simulated_voltage=simulated_voltage;
            reading.simulated_current=simulated_current;
            reading.panel_id=panel_id;
            reading.vmin=vmin;
            reading.vmax=vmax;
            reading.isc=isc;
            reading.voc=voc;
            reading.mp=mp;
            reading.imp=imp;
            reading.fillfac=fillfac;
            reading.efficiency=efficiency;
            reading.rs=rs;
            reading.rp=rp;
            reading.idark=idark;
            reading.iph=iph;
            reading.ideality=ideality;
            reading.point_count = length(measured_current);
            % Calculate avarage spacing
            reading.voltage_step = mean(abs(diff(measured_voltage)));
            reading.voltage_step_variation = std(abs(diff(measured_voltage)));
        end
        function plot(reading)
            plot(reading.conditioned_voltage, reading.conditioned_current);
            hold on;
            plot(reading.measured_voltage, reading.measured_current);
            hold on;
            plot(reading.simulated_voltage, reading.simulated_current);

        end
        function reading=circuit_parameters(reading)
            reading.isc = max(reading.measured_current([1 end]));
            reading.voc = min ([max(reading.measured_voltage), max(reading.conditioned_voltage)]);
            P = reading.measured_voltage .* reading.measured_current;
            reading.mp = max(P);
            reading.imp = reading.measured_current(P == max(P));
            reading.vmp = reading.mp / reading.imp;
            reading.fillfac = reading.mp / (reading.isc * reading.voc);
            reading.efficiency =  reading.mp / reading.irradiance/(1.65*0.992);
            c = 60 ; A = (1.65*0.992)*1e4;
            reading.cell_voc = reading.voc / c;
            reading.cell_jsc = reading.isc*1000*c/A;
            reading.cell_jmp = reading.imp*1000*c/A;
            reading.cell_vmp = reading.vmp / c;
        end
        function reading=extraction_error(reading)
            if not(isempty(reading.rs))
                if max(reading.conditioned_voltage)/max(reading.measured_voltage) <=1.2 % checking the validity of conditioned_voltage
                    voltage = linspace(0, max(reading.conditioned_voltage*0.99), 100);
                    current = interp1(reading.conditioned_voltage, reading.conditioned_current,voltage);
                    new_current = interp1(reading.simulated_voltage, reading.simulated_current, voltage);
                    nrmse = sqrt(sum((current-new_current).^2)/sum((current).^2));
                    reading.error_extraction=nrmse;
                    reading.valid_conditioning = true;

                end
            else, reading.model_found = false;

            end

        end
        function reading=expected_parameters(reading)
            % industraial parameters at
            mp = 274.353961944344; isc = 9.177586512627087; voc = 38.4962675526323;
            mu_p= -0.41/100*mp; mu_voc = -0.33/100*voc; mu_isc = 0.067/100*isc;
            Ns=60; A= 1.65*0.922;
            % circuit parameters
            rp = 346.5584311357234; rs= 1.0001253183657e-6;
            iph=9.177586512627087;
            ideality = 1.623966995034193;
            idark=1.9482304930945088e-6;
            %% Taking STC parameters from Stornelli extraction
%             rp = 216; rs= 0.172;
%             iph=9.185;
%             ideality = 1.09;
%             idark=1.01e-9;
            
            % Expected_power
            Irr_nabla= reading.irradiance / 1000;
            delT = reading.temperature - 25;
            reading.mp_expected = Irr_nabla*(mp + mu_p * delT);
            % Expected Efficiency
            reading.effciency_expected = reading.mp_expected / (reading.irradiance * A);
            % Expected ISc
            alpha = 1.0206;
            reading.isc_expected = (Irr_nabla .^alpha) .*(isc + mu_isc * delT);
            % Expected voc
            reading.voc_expected = voc + mu_voc * delT - 0.011901070649365119 ...
                + 1.5320837094311133 * log(Irr_nabla);
            % Expected fillfactor
            reading.fillfac_expected = reading.mp_expected / reading.isc_expected ...
                / reading.voc_expected;
            % expected rs and rp
            reading.rp_expected = rp / Irr_nabla;
            Tnabla = (reading.temperature +273.15)/(25+273.15);
            reading.rs_expected = Tnabla * (1-0.217*log(Irr_nabla))*rs;
            % Solving for Idark and Iph
            q = 1.60217663e-19;                  % Electron Charge
            k = 1.380649e-23;                    % Boltzmann constant            l = ideality * Ns * k * T / q
            l = ideality * Ns * k * (reading.temperature + 273.15) / q;
            A = exp(reading.voc_expected / l) - 1;
            B = exp(reading.isc_expected * reading.rs_expected / l) - 1;
            C = reading.voc_expected / reading.rp_expected;
            D = reading.isc_expected * reading.rs_expected / reading.rp_expected;
            idark = (reading.isc_expected - (C - D)) / (A - B);
            iph = idark * A + C;
            reading.idark_expected=idark;
            reading.iph_expected=iph;

        end
        function reading=validate(reading)
            reading.valid_error = reading.error_extraction <= 0.05;
            reading.valid_isc = reading.isc <= reading.isc_expected;
            reading.valid_voc = reading.voc <= reading.voc_expected;
            reading.valid_fillfac = reading.fillfac <= reading.fillfac_expected;
            reading.valid_efficiency = reading.efficiency <= reading.effciency_expected;
        end
        function reading=characterise(reading)
            %% Approximating parameters
            rs = reading.rs_expected;
            rp = reading.rp_expected;
            idark = reading.idark_expected;
            iph = reading.iph_expected;
            ideality = reading.ideality;
            V = linspace(0, reading.vmax, 100);
            q = 1.60217663e-19;
            T = (reading.temperature+273.15);
            k = 1.380649e-23;
            %% Solving equations
            syms I
            current_illuminated = zeros(size(V));
            for i = 1: length(V)
                v = V(i);
                eqn = iph - idark *  exp(q/(ideality*60*k*T)*(v + I * rs)) - (v + I * rs)/rp ==I;
                S = vpasolve(eqn, I);
                if ~isempty(S)
                    current_illuminated(i) = double(S(1));
                end
            end
            reading.voltage_expected = V;
            reading.current_expected = current_illuminated;
            %% Finding Expectation Error
            voltage = linspace(0, max(reading.conditioned_voltage), 1000);
            current = interp1(reading.conditioned_voltage, reading.conditioned_current,voltage);
            new_current = interp1(reading.voltage_expected, reading.current_expected, voltage);
            nrmse = sqrt(sum((current-new_current).^2)/sum((current).^2));
            reading.error_expectation=nrmse;

        end
        function reading=stornelli(reading)
            Isc = reading.isc; % Short circuit current
            Voc = max(reading.conditioned_voltage); %Open circuit voltage
            Imp = reading.imp; %Maximum power current
            Vmp = reading.vmp; %Maximum power voltage
            N = 60; %number of cells connected in series

            %% Starting Analysis
            Pmax = Vmp*Imp; %Maximum power point
            A = 1;
            k = 1.380649e-23;
            q = 1.60217663e-19;
            T = reading.temperature + 273.15;
            Vt = (k*A*T*N)/q;
            Rs = (Voc/Imp) - (Vmp/Imp) + ((   Vt/Imp)*log((   Vt)/(   Vt + Vmp)));
            I0 = Isc/(exp(Voc/   Vt) - exp(Rs*Isc/   Vt));
            Ipv = I0*((exp(Voc/   Vt)) - 1);
            %% First step
            iter = 10e3;
            it = 0;
            tol = 0.1;
            A1 = A;
            VmpC = (   Vt*(log((Ipv+I0-Imp)/I0))) - (Rs*Imp);
            e1 = VmpC - Vmp;
            Rs1 = Rs;
            while (it < iter & e1 > tol)
                if VmpC < Vmp
                    A1 = A1 - 0.01;
                else
                    A1 = A1 + 0.01;
                end
                Vt1 = (k*A1*T*N)/q;
                I01 = Isc/(exp(Voc/   Vt1) - exp(Rs1*Isc/   Vt1));
                Ipv1 = I01*((exp(Voc/   Vt1)) - 1);
                VmpC = (   Vt1*(log((Ipv1 + I01 - Imp)/I01))) - (Rs1*Imp);
                e1 = (VmpC - Vmp);
                it = it + 1;
            end
            Vt1 = (k*A1*T*N)/q;
            Rs1 = (Voc/Imp) - (VmpC/Imp) + ((   Vt1/Imp)*log((   Vt1)/(   Vt1 + VmpC)));
            %% Second step
            tolI = 0.001;
            iter = 10000;
            itI = 0;
            I01 = Isc/(exp(Voc/   Vt1) - exp(Rs1*Isc/   Vt1));
            Ipv1 = I01*((exp(Voc/   Vt1))-1);
            Rp = (( - Vmp)*(Vmp + (Rs1*Imp)))/(Pmax - (Vmp*Ipv1) + (Vmp*I01*(exp(((Vmp + (Rs1*Imp))/   Vt1) - 1))));
            %calculate I0 with new Rp value
            I02 = (Isc*(1 + Rs1/Rp) - Voc/Rp)/(exp(Voc/   Vt1) - exp(Rs1*Isc/   Vt1));
            Ipv2 = I02*((exp(Voc/   Vt1)) - 1) + Voc/Rp;
            ImpC = Pmax/VmpC;
            err = abs(Imp - ImpC);
            Rpnew = Rp;
            while err>tolI & itI<iter
                if ImpC<Imp
                    Rpnew = Rp + 0.1*itI;
                elseif ImpC>=Imp
                    Rpnew = Rp - 0.1*itI;
                end
                %Calculate I0 with Rpnew
                I02 = (Isc*(1 + Rs1/Rpnew) - Voc/Rpnew)/(exp(Voc/   Vt1) - exp(Rs1*Isc/   Vt1));
                Ipv2 = I02*((exp(Voc/   Vt1)) - 1) + Voc/Rpnew;
                eqn = @(ImpC) Ipv2 - (I02*(exp((Vmp + (Rs1*ImpC))/   Vt1) - 1)) - ImpC - (Vmp + Rs1*ImpC)/Rpnew;
                current_c = Imp;
                s = fzero(eqn,current_c);
                ImpC = s;
                itI = itI+1;
                err = abs(Imp - ImpC);
            end
%             X = sprintf( 'A = %.2f, I0 = %d, Ipv = %.3f, Rs = %f, Rp = %f ', A1,I02,Ipv2,Rs1,Rpnew);
%             disp(X);

            %% Plotting the extracted curve
            %A1,I02,Ipv2,Rs1,Rpnew
            ideality = abs(A1); idark=abs(I02); iph=abs(Ipv2); rs=abs(Rs1); rp=abs(Rpnew);

            V = linspace(0, Voc, 100);
            q = 1.60217663e-19;
            T = (298);
            k = 1.380649e-23;
            %---------Solving equations
            syms I
            current_illuminated = zeros(size(V));
            for i = 1: length(V)
                v = V(i);
                eqn = iph - idark *  exp(q/(ideality*N*k*T)*(v + I * rs)) - (v + I * rs)/rp ==I;
                S = vpasolve(eqn, I);
                if ~isempty(S)
                    current_illuminated(i) = double(S(1));
                end
            end
            %% Reporting output variables
            reading.stornelli_rp = rp;
            reading.stornelli_rs = rs;
            reading.stornelli_idark = idark;
            reading.stornelli_iph = iph;
            reading.stornelli_voltage = V;
            reading.stornelli_ideality = ideality;
            reading.stornelli_current = current_illuminated;
            %% Calculating stornelli error
            voltage = linspace(0, max(reading.conditioned_voltage), 100);
            current = interp1(reading.conditioned_voltage, reading.conditioned_current,voltage);
            new_current = interp1(reading.stornelli_voltage, reading.stornelli_current, voltage);
            nrmse = sqrt(sum((current-new_current).^2)/sum((current).^2));
            reading.stornelli_error=nrmse;
        end
    end
end
