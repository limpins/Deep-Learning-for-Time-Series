% figure connectivity
close all;
figure(1)
set(gcf, 'Position', 2 * [232/2   20   360   540], 'color', 'w'); % default  [232   246   560   420]
root = './seizure/';
Threld = 0.140; %

for situation = 1:3
    abcd = 13; ccc = 2; aaa = 0.85; bbb = 0.74; bb = 0.82; b = 0.9; % bbb=0.9; bb=0.82; b=0.74;
    fprintf('\n');

    switch (situation)
        case 1% pre-ictal for RNN-GC
            fprintf('Results for RNN-GC');
            load([root, 'WGCI_pre_ictal.mat']);

        case 2% Ictal1 for RNN-GC
            fprintf('Results for RNN-GC');
            load([root, 'WGCI_ictal1.mat']);
        case 3% Ictal2 for RNN-GC
            fprintf('Results for RNN-GC');
            load([root, 'WGCI_ictal2.mat']);
    end

    [m, n] = size(data);
    C_RNN = zeros(m, n);

    width = 0.42; height = 0.3;

    switch (situation)
        case 1
            a2 = [0.03 0.68 width height];
        case 2
            a2 = [0.03 0.35 width height];
        case 3
            a2 = [0.03 0.02 width height];
    end

    subplot('Position', a2);
    % channel1=[1 2 4 5 6 7 8 10 11 12 15 19]; % test channel Cp1,Cp4,Pp1,Pp4,Pp8,Ap2,Ap6,Dp1,Dp5,Bp1,Tp1,Fp2
    the1 = median(median(data));

    for ii = 1:m
        for jj = 1:n
            if ii == jj
                continue;
            end
            if data(ii, jj) < the1
                C_RNN(ii, jj) = 0;
            else
                C_RNN(ii, jj) = 1;
            end
        end
    end

    C_RNN.'
    % this functon is used to plot figure
    graph_plot(C_RNN.', 1); % plot graph
    y_axis = get(gca, 'Ylim');
    x_axis = get(gca, 'Xlim');

    switch (situation)
        case 1
            text((1 - aaa) * x_axis(1, 1) + aaa * x_axis(1, 2), aaa * y_axis(1, 2) + (1 - aaa) * y_axis(1, 1), '(a)', 'FontSize', abcd + 4);
            set(get(gca, 'Title'), 'String', 'Pre Ictal');
        case 2
            text((1 - aaa) * x_axis(1, 1) + aaa * x_axis(1, 2), aaa * y_axis(1, 2) + (1 - aaa) * y_axis(1, 1), '(c)', 'FontSize', abcd + 4);
            set(get(gca, 'Title'), 'String', 'Ictal 1');
        case 3
            text((1 - aaa) * x_axis(1, 1) + aaa * x_axis(1, 2), aaa * y_axis(1, 2) + (1 - aaa) * y_axis(1, 1), '(e)', 'FontSize', abcd + 4);
            set(get(gca, 'Title'), 'String', 'Ictal 2');
    end

    show_LCI = 0;

    if (show_LCI == 1)
        switch (situation)
            case 1
                tol2 = 'Pre Ictal: RNN-GC';
            case 2
                tol2 = 'Ictal 1: RNN-GC';
            case 3
                tol2 = 'Ictal 2: RNN-GC';
        end

        fprintf('\n');
        fprintf(tol2);
        fprintf('\n');
        fprintf('Case: ');
        fprintf(['Cp1 ', 'Cp4 ', 'Pp1 ', 'Pp4 ', 'Pp8 ', 'Ap2 ', 'Ap6 ', 'Dp1 ', 'Dp5 ', 'Bp1 ', 'Tp1 ', 'Fp2 ']);
        fprintf('\n Out:\t');
    end

    LCI = zeros(1, n);
    LCI_in = LCI; % 12
    LCI_out = LCI; % 12
    eps = 2;
    n1 = n - 1; % 1

    for kk = 1:n
        LCI_in(1, kk) = 1 / (n1) * sum(C_RNN(:, kk)); % In  C(:,kk)

        if (show_LCI == 1)
            fprintf(sprintf('%%1.%df\t', eps), LCI_in(1, kk));
        end
    end

    if (show_LCI == 1)
        fprintf('\n In:\t');
    end

    for kk = 1:n
        LCI_out(1, kk) = 1 / (n1) * sum(C_RNN(kk, :)); % Out C(:,kk)

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

    [L_value, Index] = sort(LCI, 'descend');
    L_indicate = zeros(1, n);

    for kk = 1:n
        if (L_value(1, kk) > Threld)
            L_indicate(1, kk) = 1; % Onset
            count_O = kk;
        elseif (L_value(1, kk) < -Threld)
            L_indicate(1, kk) = -1; % Propagation
        else
            L_indicate(1, kk) = 0; % not easy to distigush
            count_P = kk;
        end

    end


    NodeIDs = {'Cp1', 'Cp4', 'Pp1', 'Pp4', 'Pp8', 'Ap2', 'Ap6', 'Dp1', 'Dp5', 'Bp1', 'Tp1', 'Fp2'};

    ShowWeightsValue = 'off';
    % BGobj = biograph(C2,NodeIDs,'ShowWeights', ShowWeightsValue,'LayoutType','radial','EdgeType','straight','EdgeFontSize',4,'ArrowSize',2); % 'equilibrium','radial','hierarchical'
    Group_O = cell(1, count_O); Group_P = cell(1, count_P - count_O); Group_T = cell(1, n - count_P);
    fprintf('\n');
    fprintf('\n');

    switch (situation)
        case 1
            tol2 = 'Pre Ictal: RNN-GC';
        case 3
            tol2 = 'Ictal 1: RNN-GC';
        case 5
            tol2 = 'Ictal 2: RNN-GC';
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
    switch (situation)
        case 1
            a2 = [0.53 0.68 width height];
        case 2
            a2 = [0.53 0.35 width height];
        case 3
            a2 = [0.53 0.02 width height];
    end

    subplot('Position', a2);
    [m, n] = size(data);
    C2 = zeros(m, n);
    kk = 1;

    % node 8 and 10
    C2(8, :) = C_RNN(8, :); % Dp1
    C2(:, 8) = C_RNN(:, 8);
    C2(10, :) = C_RNN(10, :); % Bp1
    C2(:, 10) = C_RNN(:, 10);
    H = [1 2 3 4 6 7 10 5 8 11 12 9];
    fprintf('\n');

    for hh = 1:12
        fprintf(sprintf('%%1.%df', 3), LCI(1, H(hh)));
        fprintf('\n');
    end

    C2.'
    graph_plot(C2.', 1); % plot graph
    y_axis = get(gca, 'Ylim');
    x_axis = get(gca, 'Xlim');

    switch (situation)
        case 1
            text((1 - aaa) * x_axis(1, 1) + aaa * x_axis(1, 2), aaa * y_axis(1, 2) + (1 - aaa) * y_axis(1, 1), '(b)', 'FontSize', abcd + 4);
            set(get(gca, 'Title'), 'String', 'Pre Ictal Reduced');
        case 2
            text((1 - aaa) * x_axis(1, 1) + aaa * x_axis(1, 2), aaa * y_axis(1, 2) + (1 - aaa) * y_axis(1, 1), '(d)', 'FontSize', abcd + 4);
            set(get(gca, 'Title'), 'String', 'Ictal 1 Reduced');
        case 3
            text((1 - aaa) * x_axis(1, 1) + aaa * x_axis(1, 2), aaa * y_axis(1, 2) + (1 - aaa) * y_axis(1, 1), '(f)', 'FontSize', abcd + 4);
            set(get(gca, 'Title'), 'String', 'Ictal 2 Reduced');
    end
end
