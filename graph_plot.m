% this function is for plot connectivity figure

function graph_plot(rel, control)
    %???????
    %??????????????
    %??????????0??????1??????????0
    r_size = size(rel); %??????

    if nargin < 2%??????2??????
        control = 0;
    end

    if r_size(1) ~= r_size(2)
        disp('Wrong Input! The input must be a square matrix!'); %?????????????
        return;
    end

    len = r_size(2);
    %!echo len;
    %disp(len);
    rho = 4; %10
    r = 1/1.05^len; %????
    theta = 0:(2 * pi / len):2 * pi * (1 - 1 / len); %?0?????2*pi/len???2*pi*(1-1/len)?????len??
    [pointx, pointy] = pol2cart(theta', rho);
    theta = 0:pi / 36:2 * pi; %73??
    %!echo theta:
    %disp(theta);
    [tempx, tempy] = pol2cart(theta', r); %73???
    point = [pointx, pointy]; %6????
    %!echo point;
    %disp(point);
    hold on

    NodeIDs = {'Cp1', 'Cp4', 'Pp1', 'Pp4', 'Pp8', 'Ap2', 'Ap6', 'Dp1', 'Dp5', 'Bp1', 'Tp1', 'Fp2'};
    ShowWeightsValue = 'off'; %'off';
    BGobj = biograph(rel, NodeIDs, 'ShowWeights', ShowWeightsValue, 'LayoutType', 'radial', 'EdgeType', 'straight', 'EdgeFontSize', 4, 'ArrowSize', 2); % 'equilibrium', 'radial', 'hierarchical'

    for i = 1:len
        temp = [tempx, tempy] + [point(i, 1) * ones(length(tempx), 1), point(i, 2) * ones(length(tempx), 1)]; %73?1??1???point??????
        plot(temp(:, 1), temp(:, 2), 'k', 'LineWidth', 1); %plot(x,y),????
        %plot(point(:,1),point(:,2),'r');%plot(x,y)
        str{i} = BGobj.Nodes(i).ID;
        text(point(i, 1) - 0.3, point(i, 2), str{i});
        %??
    end

    for i = 1:len

        for j = 1:len

            if rel(i, j)%??????rel????1?????
                % link_plot(point(i,:),point(j,:),r,control);
                link_plot(point(j, :), point(i, :), r, control); % from j to i
                %???????
            end

        end

    end

    set(gca, 'XLim', [-rho - r, rho + r], 'YLim', [-rho - r, rho + r]);
    axis off
    %%
    function link_plot(point1, point2, r, control)
        %????
        temp = point2 - point1;

        if (~temp(1)) && (~temp(2))
            return;
            %??????
        end

        theta = cart2pol(temp(1), temp(2));
        [point1_x, point1_y] = pol2cart(theta, r);
        point_1 = [point1_x, point1_y] + point1;
        [point2_x, point2_y] = pol2cart(theta + (2 * (theta < pi) - 1) * pi, r);
        point_2 = [point2_x, point2_y] + point2;

        if control
            %     arrow(point_1,point_2);
            arrow(point_1, point_2, 'width', 0.1, 'Length', 6, 'BaseAngle', 60, 'Color', 'k');
        else
            plot([point_1(1), point_2(1)], [point_1(2), point_2(2)]);
        end

        %%
        % function arrow(start,stop,l)
        % %start,stop????????
        % %l???????????????1/10
        % % t=0.01; %
        % % ang=15/180*pi;
        % % temp=stop(1)-start(1)+1i*(stop(2)-start(2));
        % % L=abs(temp);P=angle(temp);
        % %
        % % if nargin<3
        % %     l=t*L;
        % % end
        % % p1=P-ang;p2=P+ang;
        % % a=[stop(1)-l*cos(p1) stop(2)-l*sin(p1)];
        % % b=[stop(1)-l*cos(p2) stop(2)-l*sin(p2)];
        % % hold on
        % % plot([start(1) stop(1)],[start(2) stop(2)],'k');
        % % plot([a(1) stop(1)],[a(2) stop(2)],'k');
        % % plot([b(1) stop(1)],[b(2) stop(2)],'k');
        % a=l;
        % quiver(start,stop,LineSpec,'filled');
