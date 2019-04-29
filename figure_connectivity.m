% figure connectivity
close all;
figure(16)
set(gcf, 'Position', 2 * [232/2   20   360   540], 'color', 'w'); % default  [232   246   560   420]

for situation = 2:2:6
    Threld = 0.140; %
    abcd = 13; ccc = 2; aaa = 0.85; bbb = 0.74; bb = 0.82; b = 0.9; % bbb=0.9; bb=0.82; b=0.74;
    fprintf('\n');

    switch (situation)
        case 1% pre-ictal for cPBM
            fprintf('Results for cPBM');
            load('seizure_1Table_3DCM_all_channel_pre_Ictal_pattern_selection_of_3DCM_RL1.mat');

        case 2% pre-ictal for PBM
            fprintf('Results for PBM');
            load('seizure_1Table_3DCM_all_channel_pre_Ictal_pattern_selection_of_3DCM_PBM.mat');

        case 3% Ictal1 for cPBM
            seizure = 1;
            fprintf('Results for cPBM');
            tol = ['seizure_', sprintf('%d', seizure), 'Table_3DCM_all_channel_Ictal1_pattern_selection_of_3DCM_RL1.mat'];
            load(tol);
        case 4% Ictal1 for PBM
            seizure = 1;
            fprintf('Results for PBM');
            tol = ['seizure_', sprintf('%d', seizure), 'Table_3DCM_all_channel_Ictal1_pattern_selection_of_3DCM_PBM.mat'];
            load(tol);
        case 5% Ictal2 for cPBM
            seizure = 1;
            fprintf('Results for cPBM');
            tol = ['seizure_', sprintf('%d', seizure), 'Table_3DCM_all_channel_Ictal2_pattern_selection_of_3DCM_RL1.mat'];
            load(tol);
        case 6% Ictal2 for PBM
            seizure = 1;
            fprintf('Results for PBM');
            tol = ['seizure_', sprintf('%d', seizure), 'Table_3DCM_all_channel_Ictal2_pattern_selection_of_3DCM_PBM.mat'];
            load(tol);
        case 7% Ictal for cPBM
            fprintf('Results for cPBM');
            load('seizure_1Table_3DCM_all_channel_2_Ictals_pattern_selection_of_3DCM_RL1.mat'); %cPBM

        case 8% Ictal for PBM
            fprintf('Results for PBM');
            load('seizure_1Table_3DCM_all_channel_2_Ictals_pattern_selection_of_3DCM_PBM.mat');
    end

    %     if(situation==11)
    %          fprintf('\n');
    %           fprintf('Results for cPBM');
    %
    %       %  load('seizure_1Table_3DCM_all_channel_pre_Ictal_pattern_selection_of_3DCM_RL1.mat');
    %         % load('seizure_1Table_3DCM_all_channel_pre_Ictal_pattern_selection_of_3DCM_PBM.mat');
    %           load('seizure_1Table_3DCM_all_channel_2_Ictals_pattern_selection_of_3DCM_RL1.mat'); %cPBM
    %     else
    %         fprintf('\n');
    %           fprintf('Results for PBM');
    %        % load('seizure_1Table_3DCM_all_channel_2_Ictals_pattern_selection_of_3DCM_RL1.mat');
    %          load('seizure_1Table_3DCM_all_channel_2_Ictals_pattern_selection_of_3DCM_PBM.mat');
    %     end
    [m, n] = size(ShownTable);
    C_DCM = zeros(m, n);

    for i = 1:3
        width = 0.42; height = 0.3;

        switch (1 + 2 * i - 2)
            case 1
                a2 = [0.03 0.68 width height];
            case 2
                a2 = [0.53 0.68 width height];
            case 3
                a2 = [0.03 0.35 width height];
            case 4
                a2 = [0.53 0.35 width height];
            case 5
                a2 = [0.03 0.02 width height];
            case 6
                a2 = [0.53 0.02 width height];
        end

        subplot('Position', a2);
        % channel1=[1 2 4 5 6 7 8 10 11 12 15 19]; % test channel Cp1,Cp4,Pp1,Pp4,Pp8,Ap2,Ap6,Dp1,Dp5,Bp1,Tp1,Fp2
        for ii = 1:m

            for jj = ii + 1:n

                if (i == 1)
                    [~, C_DCM(ii, jj)] = max(ShownTable{ii, jj}.Pattern{1, 3}.median_free_energy); % L-DCM  Family identification
                elseif (i == 2)
                    [~, C_DCM(ii, jj)] = max(ShownTable{ii, jj}.Pattern{1, 2}.median_free_energy); % D-DCM Family identification
                else
                    [~, C_DCM(ii, jj)] = max(ShownTable{ii, jj}.Pattern{1, 1}.median_free_energy); % DCM Family identification
                end

            end

        end

        %  figure_plot(C_DCM);
        %------------------------------------------------------------------
        [m, n] = size(C_DCM);
        C1 = zeros(m, n);

        for ii = 1:m

            for jj = ii + 1:n

                switch (C_DCM(ii, jj))
                    case 1
                        C1(ii, jj) = 0;
                        C1(jj, ii) = 0;
                    case 2
                        C1(ii, jj) = 0;
                        C1(jj, ii) = 1;
                    case 3
                        C1(ii, jj) = 1;
                        C1(jj, ii) = 0;
                    case 4
                        C1(ii, jj) = 1; % 1
                        C1(jj, ii) = 1; % 1
                end

            end

        end

        %  figure_plot(C1);
        % this functon is used to plot figure
        graph_plot(C1, 1); % plot graph
        y_axis = get(gca, 'Ylim');
        x_axis = get(gca, 'Xlim');

        switch (1 + 2 * i - 2)
            case 1
                text((1 - aaa) * x_axis(1, 1) + aaa * x_axis(1, 2), aaa * y_axis(1, 2) + (1 - aaa) * y_axis(1, 1), '(a)', 'FontSize', abcd + 4);
                %   xlabel('(a) Scenario 1 for $L-DCM$','Interpreter','latex','FontSize',abcd);
            case 2
                % title('Reslsuts for cPBM','FontSize',abcd+2);
                text((1 - aaa) * x_axis(1, 1) + aaa * x_axis(1, 2), aaa * y_axis(1, 2) + (1 - aaa) * y_axis(1, 1), '(b)', 'FontSize', abcd + 4);
            case 3
                text((1 - aaa) * x_axis(1, 1) + aaa * x_axis(1, 2), aaa * y_axis(1, 2) + (1 - aaa) * y_axis(1, 1), '(c)', 'FontSize', abcd + 4);
            case 4
                text((1 - aaa) * x_axis(1, 1) + aaa * x_axis(1, 2), aaa * y_axis(1, 2) + (1 - aaa) * y_axis(1, 1), '(d)', 'FontSize', abcd + 4);
            case 5
                text((1 - aaa) * x_axis(1, 1) + aaa * x_axis(1, 2), aaa * y_axis(1, 2) + (1 - aaa) * y_axis(1, 1), '(e)', 'FontSize', abcd + 4);
            case 6
                text((1 - aaa) * x_axis(1, 1) + aaa * x_axis(1, 2), aaa * y_axis(1, 2) + (1 - aaa) * y_axis(1, 1), '(f)', 'FontSize', abcd + 4);
        end

        show_LCI = 0;

        if (show_LCI == 1)

            switch (1 + 2 * i - 2)
                case 1
                    tol2 = 'Scenario 2: L-DCM';
                case 2
                    tol2 = 'Scenario 2: L-DCM (Reduce)';
                case 3
                    tol2 = 'Scenario 2: D-DCM';
                case 4
                    tol2 = 'Scenario 2: D-DCM (Reduce)';
                case 5
                    tol2 = 'Scenario 2: DCM';
                case 6
                    tol2 = 'Scenario 2: DCM (Reduce)';
            end

            fprintf('\n');
            fprintf(tol2);
            fprintf('\n');
            fprintf('Case: ');
            fprintf(['Cp1 ', 'Cp4 ', 'Pp1 ', 'Pp4 ', 'Pp8 ', 'Ap2 ', 'Ap6 ', 'Dp1 ', 'Dp5 ', 'Bp1 ', 'Tp1 ', 'Fp2 ']);
            fprintf('\n Out:\t');
        end

        LCI = zeros(1, n);
        LCI_in = LCI;
        LCI_out = LCI;
        eps = 2;
        n1 = n - 1; % 1

        for kk = 1:n
            LCI_in(1, kk) = 1 / (n1) * sum(C1(:, kk)); % In  C(:,kk)

            if (show_LCI == 1)
                fprintf(sprintf('%%1.%df\t', eps), LCI_in(1, kk));
            end

        end

        if (show_LCI == 1)
            fprintf('\n In:\t');
        end

        for kk = 1:n
            LCI_out(1, kk) = 1 / (n1) * sum(C1(kk, :)); % Out C(:,kk)

            if (show_LCI == 1)
                fprintf(sprintf('%%1.%df\t', eps), LCI_out(1, kk));
            end

        end

        if (show_LCI == 1)
            fprintf('\n All:\t');
        end

        for kk = 1:n
            %LCI(1,kk)=(LCI_in(1,kk)-LCI_out(1,kk)); % C(:,kk) - C(kk,:)

            LCI(1, kk) = ((LCI_in(1, kk) - LCI_out(1, kk))) / ((LCI_in(1, kk) + LCI_out(1, kk))); % C(:,kk) - C(kk,:)

            if (show_LCI == 1)
                fprintf(sprintf('%%1.%df\t', eps), LCI(1, kk));
            end

        end

        % for new

        % channel1=[1 2 4 5 6 7 8 10 11 12 15 19]; % test channel Cp1,Cp4,Pp1,Pp4,Pp8,Ap2,Ap6,Dp1,Dp5,Bp1,Tp1,Fp2

        %           fprintf('\n');
        %         % fprintf(L_value(1,1:end));
        %          for lll=1:12
        %          fprintf(sprintf('%%1.%df',2),L_value(1,lll));
        %          fprintf('\t');
        %          end
        %         fprintf('\n');

        [L_value, Index] = sort(LCI, 'descend');

        L_indicate = zeros(1, n);

        for kk = 1:n

            if (L_value(1, kk) > Threld)
                L_indicate(1, kk) = 1; % Onset
                count_O = kk;
            elseif (L_value(1, kk) <- Threld)
                L_indicate(1, kk) = -1; % Propagation
            else
                L_indicate(1, kk) = 0; % not easy to distigush
                count_P = kk;
            end

        end

        NodeIDs = {'Cp1', 'Cp4', 'Pp1', 'Pp4', 'Pp8', 'Ap2', 'Ap6', 'Dp1', 'Dp5', 'Bp1', 'Tp1', 'Fp2'};
        %         figure(10000)
        %         plot (LCI);
        % xlabel(NodeIDs);
        %        set(gca,'XtickLabel',NodeIDs{1,1:12});

        ShowWeightsValue = 'off';
        % BGobj = biograph(C2,NodeIDs,'ShowWeights', ShowWeightsValue,'LayoutType','radial','EdgeType','straight','EdgeFontSize',4,'ArrowSize',2); % 'equilibrium','radial','hierarchical'
        Group_O = cell(1, count_O); Group_P = cell(1, count_P - count_O); Group_T = cell(1, n - count_P);
        fprintf('\n');
        fprintf('\n');

        switch (1 + 2 * i - 2)
            case 1
                tol2 = 'Scenario 2: L-DCM';
            case 3
                tol2 = 'Scenario 2: D-DCM';
            case 5
                tol2 = 'Scenario 2: DCM';
        end

        fprintf(tol2);
        fprintf('\n');
        fprintf('Group O:\t');

        for kk = 1:count_O
            Tol = NodeIDs{1, Index(1, kk)};
            Group_O{1, kk} = Tol;
            fprintf(Tol);
            fprintf(',\t');
        end

        fprintf('\n');
        fprintf('Group PI:\t');

        for kk = count_O + 1:count_P
            Tol = NodeIDs{1, Index(1, kk)};
            Group_P{1, kk - count_O} = Tol;
            fprintf(Tol);
            fprintf(',\t');
        end

        fprintf('\n');
        fprintf('Group PT:\t');

        for kk = count_P + 1:n
            Tol = NodeIDs{1, Index(1, kk)};
            Group_T{1, kk - count_P} = Tol;
            fprintf(Tol);
            fprintf(',\t');
        end

        fprintf('\n');
        % fprintf(L_value(1,1:end));
        for lll = 1:12
            fprintf(sprintf('%%1.%df', 2), L_value(1, lll));
            fprintf('\t');
        end

        fprintf('\n');
        % considering reduced model
        switch (2 + 2 * i - 2)
            case 1
                a2 = [0.03 0.68 width height];
            case 2
                a2 = [0.53 0.68 width height];
            case 3
                a2 = [0.03 0.35 width height];
            case 4
                a2 = [0.53 0.35 width height];
            case 5
                a2 = [0.03 0.02 width height];
            case 6
                a2 = [0.53 0.02 width height];
        end

        subplot('Position', a2);
        [m, n] = size(C_DCM);
        C2 = zeros(m, n);
        kk = 1;
        %          %============= maximum two nodes
        %         for ii=1:m
        %             if(ii<3||ii>10) %
        %                 if(L_indicate(1,ii)>0)   % onset group
        %                     C2(Index(1,ii),:)=C1(Index(1,ii),:);
        %                     C2(:,Index(1,ii))=C1(:,Index(1,ii));
        %                 end
        %                 if(L_indicate(1,ii)<0)   % Pt group
        %                     C2(Index(1,ii),:)=C1(Index(1,ii),:);
        %                     C2(:,Index(1,ii))=C1(:,Index(1,ii));
        %                 end
        %             end
        %         end % for ii

        % node 8 and 10
        C2(8, :) = C1(8, :); % Dp1
        C2(:, 8) = C1(:, 8);
        C2(10, :) = C1(10, :); % Bp1
        C2(:, 10) = C1(:, 10);
        H = [1 2 3 4 6 7 10 5 8 11 12 9];
        fprintf('\n');

        for hh = 1:12

            fprintf(sprintf('%%1.%df', 3), LCI(1, H(hh)));
            fprintf('\n');

        end

        % figure_plot(C_2);
        graph_plot(C2, 1); % plot graph
        y_axis = get(gca, 'Ylim');
        x_axis = get(gca, 'Xlim');

        switch (2 + 2 * i - 2)
            case 1
                text((1 - aaa) * x_axis(1, 1) + aaa * x_axis(1, 2), aaa * y_axis(1, 2) + (1 - aaa) * y_axis(1, 1), '(a)', 'FontSize', abcd + 4);
                %   xlabel('(a) Scenario 1 for $L-DCM$','Interpreter','latex','FontSize',abcd);
            case 2
                % title('Reslsuts for cPBM','FontSize',abcd+2);
                text((1 - aaa) * x_axis(1, 1) + aaa * x_axis(1, 2), aaa * y_axis(1, 2) + (1 - aaa) * y_axis(1, 1), '(b)', 'FontSize', abcd + 4);
            case 3
                text((1 - aaa) * x_axis(1, 1) + aaa * x_axis(1, 2), aaa * y_axis(1, 2) + (1 - aaa) * y_axis(1, 1), '(c)', 'FontSize', abcd + 4);
            case 4
                text((1 - aaa) * x_axis(1, 1) + aaa * x_axis(1, 2), aaa * y_axis(1, 2) + (1 - aaa) * y_axis(1, 1), '(d)', 'FontSize', abcd + 4);
            case 5
                text((1 - aaa) * x_axis(1, 1) + aaa * x_axis(1, 2), aaa * y_axis(1, 2) + (1 - aaa) * y_axis(1, 1), '(e)', 'FontSize', abcd + 4);
            case 6
                text((1 - aaa) * x_axis(1, 1) + aaa * x_axis(1, 2), aaa * y_axis(1, 2) + (1 - aaa) * y_axis(1, 1), '(f)', 'FontSize', abcd + 4);
        end

        % onset group propagation group
        %         hold on;
        %         NodeIDs = {'Cp1', 'Cp4', 'Pp1', 'Pp4', 'Pp8', 'Ap2', 'Ap6','Dp1', 'Dp5', 'Bp1','Tp1','Fp2'};
        %         ShowWeightsValue='off';%'off';
        %         BGobj = biograph(C1,NodeIDs,'ShowWeights', ShowWeightsValue,'LayoutType','radial','EdgeType','straight','EdgeFontSize',4,'ArrowSize',2); % 'equilibrium','radial','hierarchical'
        %         set(BGobj.Nodes(1:12),'shape','circle','lineColor',[0, 0, 0],'LineWidth',1); %ellipse
        %         dolayout(BGobj);
        %         dolayout(BGobj,'Pathsonly', true);
        %         h=view(BGobj);
        %         %set(h.Nodes(1:12),'shape','circle','lineColor',[0, 0, 0],'LineWidth',1); %ellipse
        %        % [S, C] = biograph.conncomp(C1);
        %        %  set(h.Nodes(1:12),'shape','circle','lineColor',[0, 0, 0],'LineWidth',1); %ellipse
        % %------------------------------------------------------------------
        % % this functon is used to plot figure
        %                 graph_plot(C1,1); % plot graph
        %                 y_axis=get(gca,'Ylim');
        %                 x_axis=get(gca,'Xlim');
        %                 switch(situation-9+2*i-2)
        %                     case 1
        %                        text((1-aaa)*x_axis(1,1)+aaa*x_axis(1,2),aaa*y_axis(1,2)+(1-aaa)*y_axis(1,1),'(a)','FontSize',abcd+4);
        %                       %   xlabel('(a) Scenario 1 for $L-DCM$','Interpreter','latex','FontSize',abcd);
        %                     case 2
        %                            % title('Reslsuts for cPBM','FontSize',abcd+2);
        %                         text((1-aaa)*x_axis(1,1)+aaa*x_axis(1,2),aaa*y_axis(1,2)+(1-aaa)*y_axis(1,1),'(b)','FontSize',abcd+4);
        %                     case 3
        %                         text((1-aaa)*x_axis(1,1)+aaa*x_axis(1,2),aaa*y_axis(1,2)+(1-aaa)*y_axis(1,1),'(c)','FontSize',abcd+4);
        %                     case 4
        %                         text((1-aaa)*x_axis(1,1)+aaa*x_axis(1,2),aaa*y_axis(1,2)+(1-aaa)*y_axis(1,1),'(d)','FontSize',abcd+4);
        %                     case 5
        %                         text((1-aaa)*x_axis(1,1)+aaa*x_axis(1,2),aaa*y_axis(1,2)+(1-aaa)*y_axis(1,1),'(e)','FontSize',abcd+4);
        %                     case 6
        %                         text((1-aaa)*x_axis(1,1)+aaa*x_axis(1,2),aaa*y_axis(1,2)+(1-aaa)*y_axis(1,1),'(f)','FontSize',abcd+4);
        %                 end
    end

    clearvars - except situation;
end

set(gcf, 'Color', 'w');
