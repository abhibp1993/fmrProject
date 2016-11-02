%% Ari Goodman
%% 11/1/2016
%% Given a vector of car states Car_States, plots them on a grid world of size N

%% Inputs:
% World: Vector of objects [id, i,j,t,s] where id is the classifier, i is the x position, j is the y
% position, t is the angle(0,1,2,3), and s is speed in cells/timestep
% N: grid size in cells. Optional.

%% Outputs:
% Plot of cars.

function plotWorld(World, N)
    if nargin == 0
        disp('Error. Please input world.');
        return
    elseif nargin ==1
        N = 20;
    end
    cars = [];
    for obj = 1:size(World,2)
        if isa(World(obj), 'Car')
            cars = [cars World(obj)]
        end
    end
    
    world = ones(N);
    figure;
    for obj = 1:size(cars,2)
       world(cars(obj).y,cars(obj).x) = 0; %% TODO: add in coloring for other objects
    end
    h = pcolor(world);
    
    for obj = 1:size(cars,2)
        car = cars(obj);
        text(car.x+.5,car.y+.5,1,sprintf('%d',car.id),'Color','white','FontSize',10);
    end
    colormap(gray(2));
   % axis ij
    axis square
    set(h, 'EdgeColor', [.8 .8 .8]);
    hold on;
        
    function x = get_x(direction)
        for d = 1:size(direction,1)
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
        for d = 1:size(direction,1)
            if direction(d) == 0
                y(d)=1;
            elseif direction(d) == 2
                y(d)=-1;
            else
                y(d)=0;
            end
        end
    end
    for car_iterator = 1:size(cars,2)
        car = cars(car_iterator);
        dir_car(car_iterator,1) = car.speed.*get_x(car.h)';
        dir_car(car_iterator,2) = car.speed.*get_y(car.h)';
        x_car(car_iterator) = car.x+.5;
        y_car(car_iterator) = car.y+.5;
    end
    quiver(x_car(:),y_car(:),dir_car(:,1), dir_car(:,2), 1/2, 'linewidth', 2);
    %% TODO: Fix quiver
    
    %% Add key for cars and other objects
    title('World Plot of Cars');
end
