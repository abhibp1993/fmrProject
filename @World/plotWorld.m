%% Ari Goodman
%% 11/2/2016
%% Given a vector of car states Car_States, plots them on a grid world of size N

%% Inputs:
% World: Vector of objects [id, i,j,t,s] where id is the classifier, i is the x position, j is the y
% position, t is the angle(0,1,2,3), and s is speed in cells/timestep
% N: grid size in cells. Optional.

%% Outputs:
% Plot of cars.

function plotWorld(World)
   
    keySet =   {'Car', 'Terrain', 'Road', 'Hazard', 'Yield', 'Stop', 'RED', 'YELLOW', 'GREEN'};
    CarColor =      {[.1,.1,.4]};
    TerrainColor =  {[.1,.4,.1]};
    RoadColor =     {[0,0,0]};
    HazardColor =   {[1.,.5,0]};
    YieldColor =    {[.9,.9,0]};
    StopColor =     {[.9,0,0]};
    RedColor =      {[1,0,0]};
    YellowColor =   {[1,1,0]};
    GreenColor =    {[0,1,0]};
    
    valueSet = [CarColor,TerrainColor,RoadColor,HazardColor,YieldColor,StopColor,RedColor,YellowColor,GreenColor];
    colorMap = containers.Map(keySet,valueSet);
   
    cars = {};
    trafficSigns = {};
    trafficLights = {};
    roads = {};
    hazards = {};
    %TODO: make more efficient
    for obj_i = 1:size(World.objects,1)
        for obj_j = 1:size(World.objects,2)
            if(length(World.objects{obj_i,obj_j}) == 1)
                if isa(World.objects{obj_i,obj_j}, 'SDC') || isa(World.objects{obj_i,obj_j}, 'HDC') %determine type of object
                    cars = [cars World.objects{obj_i,obj_j}];
                elseif isa(World.objects{obj_i,obj_j}, 'TrafficSign')
                    trafficSigns = [trafficSigns World.objects{obj_i,obj_j}];
                elseif isa(World.objects{obj_i,obj_j}, 'TrafficLight')
                    trafficLights = [trafficLights World.objects{obj_i,obj_j}];
                elseif isa(World.objects{obj_i,obj_j}, 'Road')
                    roads = [roads World.objects{obj_i,obj_j}];
                elseif isa(World.objects{obj_i,obj_j}, 'Hazard')
                    hazards = [hazards World.objects{obj_i,obj_j}];
                end
            else
                for obj_k = 1:length(World.objects{obj_i,obj_j})
                    if isa(World.objects{obj_i,obj_j}{obj_k}, 'SDC') || isa(World.objects{obj_i,obj_j}{obj_k}, 'HDC') %determine type of object
                        cars = [cars World.objects{obj_i,obj_j}{obj_k}];
                    elseif isa(World.objects{obj_i,obj_j}{obj_k}, 'TrafficSign')
                        trafficSigns = [trafficSigns World.objects{obj_i,obj_j}{obj_k}];
                    elseif isa(World.objects{obj_i,obj_j}{obj_k}, 'TrafficLight')
                        trafficLights = [trafficLights World.objects{obj_i,obj_j}{obj_k}];
                    elseif isa(World.objects{obj_i,obj_j}{obj_k}, 'Road')
                        roads = [roads World.objects{obj_i,obj_j}{obj_k}];
                    elseif isa(World.objects{obj_i,obj_j}{obj_k}, 'Hazard')
                        hazards = [hazards World.objects{obj_i,obj_j}{obj_k}];
                    end
                end
            end
        end
    end

    figure;
    hold on;
    
    if(~isempty(roads))
         for road = 1:length(roads)
            plot(roads(road).state.x,roads(road).state.y,'s', 'MarkerEdgeColor', colorMap('Road'), 'MarkerFaceColor', colorMap('Road'), 'MarkerSize', 65) %TODO: make size adjusted based on figure window
         end
    end
    if(~isempty(cars))
        for car = 1:length(cars)
            plot(cars(car).state.x,cars(car).state.y,'o', 'MarkerFaceColor', colorMap('Car'), 'MarkerSize', 40)
        end
    end
    if(~isempty(trafficLights))
        for tl = 1:length(trafficLights)
            plot(trafficLights(tl).state.x,trafficLights(tl).state.y,'*', 'LineWidth', 2, 'Color', colorMap(trafficLights(tl).state.light),'MarkerSize', 10)
        end
    end
    if(~isempty(trafficSigns))
        for ts = 1:length(trafficSigns)
            plot(trafficSigns(ts).state.x,trafficSigns(ts).state.y,'x', 'LineWidth', 2, 'Color', colorMap(trafficSigns(ts).type),'MarkerSize', 10)
        end
    end
    
    axis([.5 World.xLength+.5 .5 World.yLength+.5]);
    set(gca, 'Color',colorMap('Terrain'));
    % add special stuff for cars, like vision and speed
     for obj = 1:length(cars)
         car = cars(obj);
         senseMask = rot90(car.senseMask,car.state.h);
         text(car.state.x,car.state.y,1,sprintf('%d',car.id),'Color','white','FontSize',20);
         for sight_i = 1:size(senseMask,1) %TODO: replace with convex hull
             for sight_j = 1:size(senseMask,2)
                 pos = [car.state.x+sight_i-size(senseMask,1)/2-.5,car.state.y+sight_j-size(senseMask,2)/2-.5];
                 if(senseMask(sight_i,sight_j) && pos(1) >= 1 && pos(2) >=1 && pos(1) <= World.xLength && pos(2) <= World.yLength)
                     text(pos(1),pos(2),'o','Color','green','Fontsize',8);
                 end
             end
         end
     end
     
%     for obj = 1:length(trafficLights) %might be necessary to distinguish yield from yellow and red from stop
%         light = trafficLights(obj);
%         text(light.state.x,light.state.y,1,'.','Color','white','FontSize',10);
%     end
   % colormap(colorcube(20));
   % axis ij
    axis square
   %set(h, 'EdgeColor', [.8 .8 .8]);
    hold on;
%         
%     function x = get_x(direction)
%         for d = 1:length(direction)
%             if direction(d) == 1
%                 x(d)=-1;
%             elseif direction(d) == 3
%                 x(d)=1;
%             else
%                 x(d)=0;
%             end
%         end
%     end
% 
%     function y = get_y(direction)
%         for d = 1:length(direction)
%             if direction(d) == 0
%                 y(d)=1;
%             elseif direction(d) == 2
%                 y(d)=-1;
%             else
%                 y(d)=0;
%             end
%         end
%     end
%     for car_iterator = 1:length(cars)
%         car = cars(car_iterator);
%         dir_car(car_iterator,1) = 1/4*car.state.speed.*get_x(car.state.h)'; %must prescale vectors due to relative scaling
%         dir_car(car_iterator,2) = 1/4*car.state.speed.*get_y(car.state.h)';
%         x_car(car_iterator) = car.state.x;
%         y_car(car_iterator) = car.state.y;
%     end
%     if length(cars) > 0
%         quiver(x_car(:),y_car(:),dir_car(:,1), dir_car(:,2), 0, 'linewidth', 2); %ensure no relative scaling
%     end
    % TODO
    %colorbar('Ticks',sort(valueSet),'TickLabels',keySet)
    %% Add key for cars and other objects
    title('World Plot of Cars');
end