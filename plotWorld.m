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
   
    keySet =   {'Car', 'Terrain', 'Road', 'Hazard', 'Yield', 'Stop', 'Red', 'Yellow', 'Green'};
    valueSet = [0, 1, 2, 3, 4, 5, 6, 7 ,8];
    colorMap = containers.Map(keySet,valueSet);
   
    cars = [];
    trafficSigns = [];
    trafficLights = [];
    roads = [];
    hazards = [];
    for obj = 1:length(World.objects)
        %if ~(World(obj).y > N || World(obj).x > N || World(obj).y <= 0 || World(obj).x <= 0) %ensure it is within legal bounds
        if isa(World.objects(obj), 'Car') %determine type of object
            cars = [cars World.objects(obj)];
        elseif isa(World.objects(obj), 'TrafficSign')
            trafficSigns = [trafficSigns World.objects(obj)];
        elseif isa(World.objects(obj), 'TrafficLight')
            trafficLights = [trafficLights World.objects(obj)];
        elseif isa(World.objects(obj), 'Road')
            roads = [roads World.objects(obj)];
        elseif isa(World.objects(obj), 'Hazard')
            hazards = [hazards World.objects(obj)];
        end
    end
    
    world = ones(N)*colorMap('Terrain');
    figure;
    for obj = 1:length(cars)
       world(cars(obj).y,cars(obj).x) = colorMap('Car'); %set coloring
    end
    for obj = 1:length(trafficSigns)
       world(trafficSigns(obj).y,trafficSigns(obj).x) = colorMap(trafficSigns(obj).type); 
    end
    for obj = 1:length(trafficLights)
       world(trafficLights(obj).y,trafficLights(obj).x) = colorMap(trafficLights(obj).color);
    end
    for obj = 1:length(roads)
       world(roads(obj).y,roads(obj).x) = colorMap('Road');
    end
    for obj = 1:length(hazards)
       world(hazards(obj).y,hazards(obj).x) = colorMap('Hazard');
    end
    h = imagesc(world);
    set(gca,'YDir','normal')
    for obj = 1:length(cars)
        car = cars(obj);
        senseMask = rot90(car.senseMask,car.h);
        text(car.x,car.y,1,sprintf('%d',car.id),'Color','white','FontSize',10);
        for sight_i = 1:size(senseMask,1)
            for sight_j = 1:size(senseMask,1)
                if(senseMask(sight_i,sight_j))
                    text(car.x+sight_i-size(senseMask,1)/2-.5,car.y+sight_j-size(senseMask,2)/2-.5,'o','Color','green','Fontsize',8);
                end
            end
        end
    end
    for obj = 1:length(trafficLights) %might be necessary to distinguish yield from yellow and red from stop
        light = trafficLights(obj);
        text(light.x,light.y,1,'.','Color','white','FontSize',10);
    end
    colormap(gray(10));
   % axis ij
    axis square
   %set(h, 'EdgeColor', [.8 .8 .8]);
    hold on;
        
    function x = get_x(direction)
        for d = 1:length(direction)
            if direction(d) == 1
                x(d)=-1;
            elseif direction(d) == 3
                x(d)=1;
            else
                x(d)=0;
            end
        end
    end

    function y = get_y(direction)
        for d = 1:length(direction)
            if direction(d) == 0
                y(d)=1;
            elseif direction(d) == 2
                y(d)=-1;
            else
                y(d)=0;
            end
        end
    end
    for car_iterator = 1:length(cars)
        car = cars(car_iterator);
        dir_car(car_iterator,1) = 1/4*car.speed.*get_x(car.h)'; %must prescale vectors due to relative scaling
        dir_car(car_iterator,2) = 1/4*car.speed.*get_y(car.h)';
        x_car(car_iterator) = car.x;
        y_car(car_iterator) = car.y;
    end
    quiver(x_car(:),y_car(:),dir_car(:,1), dir_car(:,2), 0, 'linewidth', 2); %ensure no relative scaling
    colorbar('Ticks',valueSet,'TickLabels',keySet)
    %% Add key for cars and other objects
    title('World Plot of Cars');
end